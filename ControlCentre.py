"""UI
remote-control: tello-sdk
voice-interaction: speak
compute-engine
    image-processing (pre&post)
    model-inference
visualization:
    inference-result
    tello-status
"""

import os
import time
import traceback

import cv2
import sys
import logging
import threading
import numpy as np
import pyqtgraph as pg
from collections import deque
from DataPipeline import DataPipeline
from ModelInference import PoseDetector, IkunRecognizer, FaceDetector
# from SpeechAgent import SpeechAgent
# from djitellopy import Tello
from debugtellopy import DebugTello as Tello
from PID import PID
from ControlCentreUI import Ui_MainWindow
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QThread, QDir, QTimer, QSize, Qt
from PyQt5.QtGui import QPixmap, QImage, QTextCursor
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QMessageBox, QDialog, QLineEdit, QDialogButtonBox, QFormLayout
from PyQt5.QtGui import QIntValidator, QDoubleValidator

logging.basicConfig(format='%(process)s-%(thread)d-%(levelname)s-%(message)s', level=logging.DEBUG)

# order: yaw->zpos->xpos
ORDERS = ['yaw', 'zpos', 'xpos']
SAMPLES_NUMBER = 100
RECOGNIZER_THR = 0.55


class ProcessStreamThread(QThread):
    stream_signal = QtCore.pyqtSignal(tuple)
    fps = 20
    loop_sec = 1 / fps

    def __init__(self, tello_agent, detector_config):
        super().__init__()
        self.tello_agent = tello_agent  # for frame-getting
        self.detector = PoseDetector(detector_config['model_complexity'],
                                     detector_config['enable_segmentation'],
                                     detector_config['min_detection_confidence'],
                                     detector_config['min_tracking_confidence'])
        self.stream_ret_queue = deque(maxlen=2000)
        self.isStreaming = False
        self.frame_h, self.frame_w = (None, None)

    def get_frame_shape(self):
        return self.frame_h, self.frame_w

    def run(self) -> None:

        while True:
            start_time = time.time()

            frame = self.tello_agent.get_frame_read().frame
            try:
                frame = DataPipeline.pipe_pre(frame)  # flip and convert-color
            except ValueError as e:
                traceback.print_exc()
                continue

            if not self.isStreaming:
                self.isStreaming = True
                self.frame_h, self.frame_w = frame.shape[:2]

            detect_ret = self.detector.inference(frame)  # mediapipe's output

            time_consume = time.time() - start_time
            # print(f"consume time: {time_consume * 1000}ms")

            self.stream_ret_queue.append((frame, detect_ret, time_consume))
            self.stream_signal.emit((frame, detect_ret))

            if time_consume <= self.loop_sec:
                time.sleep(self.loop_sec - time_consume)
            else:
                # logging.info("Lower FPS than set value!")
                pass


class VisualizationThread(QThread):
    visualization_signal = QtCore.pyqtSignal(dict)

    def __init__(self, recognizer_path, label_imageTransmission, label_skeleton, tello_status_dict, stream_ret_queue):
        super().__init__()
        self.label_imageTransmission = label_imageTransmission
        self.label_skeleton = label_skeleton
        self.stream_ret_queue = stream_ret_queue  # should be a shallow copy!
        self.tello_status_dict = tello_status_dict  # should be a shallow copy!

        # store pose-landmarks detection result
        self.detect_ret_list = []

        self.recognizer = IkunRecognizer(recognizer_path)

    def update_frame(self, frame):
        """Update UI-frame"""
        showImage = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(showImage)
        self.label_imageTransmission.setPixmap(pixmap)

    def update_graphic(self, skeleton_image):
        showImage = QImage(skeleton_image.data, skeleton_image.shape[1], skeleton_image.shape[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(showImage)
        self.label_skeleton.setPixmap(pixmap)

        # graphic-1: update flying curve
        # available_length = len(self.controller_input_queue)
        # self.signal_arrays[self.SAMPLES_NUMBER - available_length:] = self.controller_input_queue
        #
        # tab_index = self.tabWidget.currentIndex()
        # self.signal_plots[tab_index].setData(self.time_array, self.signal_arrays[tab_index])
        # self.signal_plots[tab_index].updateItems()
        # self.signal_plots[tab_index].sigPlotChanged.emit(self.signal_plots[tab_index])  # ?

    def run(self) -> None:
        while True:
            if self.stream_ret_queue:
                # print(len(self.stream_ret_queue))
                start_time = time.time()

                # image view
                frame, detect_ret, time_consume = self.stream_ret_queue.popleft()
                if detect_ret is not None:
                    # append the latest pose-landmarks detection result
                    self.detect_ret_list.append(detect_ret)
                    self.detect_ret_list.pop(0) if len(self.detect_ret_list) > self.recognizer.SEQUENCE_NUM else None

                if len(self.detect_ret_list) == IkunRecognizer.SEQUENCE_NUM:
                    scores = self.recognizer.inference(self.detect_ret_list)
                    argmax = np.argmax(scores)
                    is_ikun = True if (argmax == 1 and (scores[0][argmax] > RECOGNIZER_THR)) else False
                else:
                    is_ikun = False

                infer = {'landmark': detect_ret, 'is_ikun': is_ikun}

                # print(self.tello_status_dict)
                self.tello_status_dict['FPS'] = round(1/time_consume, 1)
                frame, skeleton = DataPipeline.pipe_post(frame, infer, self.tello_status_dict)
                self.update_frame(frame)
                self.update_graphic(skeleton)

                # if skeleton is not None:
                #     cv2.imshow('demo', skeleton)
                #     if cv2.waitKey(1) & 0xFF == ord('q'):
                #         break

                self.visualization_signal.emit(infer)

                time_consume = time.time() - start_time
                # print(f"visualize consume: {time_consume * 1000}ms")
                # if 'delay' is not used, there will be a very significant display delay even of '0.01ms'!
                time.sleep(max(0., ProcessStreamThread.loop_sec - time_consume))


class ControlThread(QThread):
    output_signal = QtCore.pyqtSignal(str)

    def __init__(self, tello_agent, pid_input_queue: deque):
        super().__init__()
        self.tello_agent = tello_agent  # for remote-control
        self.pid_input_queue = pid_input_queue  # for communication with main-thread
        self._auto_mode = False

        self.yaw_pid = PID(1, 0.01, 0.1)  # yaw control
        self.zpos_pid = PID(1, 0.01, 0.1)  # up-down control
        self.xpos_pid = PID(1, 0.01, 0.1)  # forward-backward control
        self.yaw_pid.output_limits = (-50, 50)
        self.zpos_pid.output_limits = (-50, 50)
        self.xpos_pid.output_limits = (-50, 50)

    @property
    def auto_mode(self):
        return self._auto_mode

    @auto_mode.setter
    def auto_mode(self, enabled):
        self.set_auto_mode(enabled)

    def set_auto_mode(self, enabled: bool):
        self._auto_mode = enabled

        # remember to set each pid-controller's auto_mode properties
        self.yaw_pid.auto_mode = enabled
        self.zpos_pid.auto_mode = enabled
        self.xpos_pid.auto_mode = enabled

    def set_controller_setpoints(self, setpoints):
        self.yaw_pid.setpoint = setpoints['yaw']
        self.zpos_pid.setpoint = setpoints['zpos']
        self.xpos_pid.setpoint = setpoints['xpos']

    def tunings(self, ctrl_type: str, values: tuple):
        if ctrl_type == 'yaw':
            self.yaw_pid.tunings(values)
        elif ctrl_type == 'zpos':
            self.zpos_pid.tunings(values)
        elif ctrl_type == 'xpos':
            self.xpos_pid.tunings(values)
        else:
            raise ValueError

    def run(self) -> None:
        while True:
            if self.tello_agent.is_flying and self._auto_mode:
                try:
                    inputs = self.pid_input_queue[-1]  # QA: pop
                except:
                    logging.warning('Not enough measurements!')
                    continue

                yaw_output = self.yaw_pid(inputs[0]) if inputs[0] is not None else 0
                zpos_output = self.zpos_pid(inputs[1]) if inputs[1] is not None else 0
                xpos_output = self.xpos_pid(inputs[2]) if inputs[2] is not None else 0

                """
                left_right_vel = 0
                forward_backward_vel = min(100, max(xpos_output, -100))
                up_down_vel = min(100, max(zpos_output, -100))
                yaw_vel = min(100, max(yaw_output, -100))  # or max(-100, min(yaw_output, 100))
                self.tello_agent.send_rc_control(left_right_vel, forward_backward_vel, up_down_vel, yaw_vel)
                """
                if not np.any((yaw_output, zpos_output, xpos_output)):
                    self.tello_agent.send_rc_control(0, xpos_output, zpos_output, yaw_output)
                else:
                    # logging.warning("")
                    pass
            else:
                pass


class TelloStatusThread(QThread):
    status_signal = QtCore.pyqtSignal(str)

    def __init__(self, tello_agent: Tello):
        super().__init__()
        self.tello_agent = tello_agent
        self.status = {}

    def run(self) -> None:
        while True:
            try:
                # can't reassign 'self.status' using this way: self.status = {...}
                # keep only those states or attributes that will not change dramatically
                # dimension: °C, %, cm, s
                self.status['temperature'] = self.tello_agent.get_temperature(),
                # 'height': self.tello_agent.get_height(),
                # 'height_barometer': self.tello_agent.get_barometer(),
                # 'distance_tof': self.tello_agent.get_distance_tof(),
                self.status['battery'] = self.tello_agent.get_battery()
                self.status['flight_time'] = self.tello_agent.get_flight_time()

                if self.status['battery'] < 20:
                    self.status_signal.emit(f"Low Battery: {self.status['battery']}%")
            except ConnectionAbortedError:
                print(traceback.format_exc())

            # update status of tello per 5-sec.
            QThread.sleep(5)


class MainWindow(QMainWindow, Ui_MainWindow):
    machine_signal = QtCore.pyqtSignal(object)

    def __init__(self, recognizer_path, yunet_path, mpose_config):
        super().__init__()
        self.setupUi(self)
        self.setupUiMore()

        # graphic-view initialization
        pen = pg.mkPen(color=(255, 92, 92), width=1.5)
        self.time_array = np.arange(SAMPLES_NUMBER, 0, 1)
        self.graphicsViews = [self.graphicsView_yaw_plotting, self.graphicsView_zpos_plotting, self.graphicsView_xpos_plotting]
        self.signal_arrays = []
        self.signal_plots = []
        for i in range(3):
            self.graphicsViews.append(eval('self.graphicsView_' + ORDERS[i] + '_plotting'))
            self.signal_arrays.append(np.zeros(SAMPLES_NUMBER))
            self.signal_plots.append(pg.PlotDataItem(self.time_array, self.signal_arrays[i], pen=pen, name=f'{ORDERS[i]} [pixel]'))
            # pyqtgraph addItem
            self.graphicsViews[i].addItem(self.signal_plots[i])

        # queue (used in different thread)
        self.controller_input_queue = deque(maxlen=SAMPLES_NUMBER)  # [(yaw_input, zpos_input, xpos_input), ...]
        self.speech_msg_queue = deque(maxlen=10)

        # constructor
        # self.agent = SpeechAgent()
        self.tello = Tello()

        self.face_detector = FaceDetector(yunet_path)

        # initialize
        self.tello.connect()

        # self.tello.set_video_resolution(Tello.RESOLUTION_480P)
        # self.tello.set_video_bitrate(Tello.BITRATE_1MBPS)
        # self.tello.set_video_fps(Tello.FPS_30)
        self.tello.streamon()

        # initialize tello status updating thread (to avoid the impact of instant wifi-communication on FPS)
        self.tello_status_tracker = TelloStatusThread(self.tello)
        # initialize processing thread
        self.stream_processor = ProcessStreamThread(self.tello, mpose_config)
        # ...
        self.visualizer = VisualizationThread(recognizer_path,
                                              self.label_imageTransmission,
                                              self.label_skeleton,
                                              self.tello_status_tracker.status,
                                              self.stream_processor.stream_ret_queue)
        # initialize position control thread
        self.position_controller = ControlThread(self.tello, self.controller_input_queue)

        # some tricks
        self.is_people = False
        self.is_ikun = False

        # register callback
        self.register_callback()

    def setupUiMore(self):
        # lineEdit validator
        lineEdit_names = ['lineEdit_YawProportional', 'lineEdit_YawIntegral', 'lineEdit_YawDerivative',
                          'lineEdit_ZPosProportional', 'lineEdit_ZPosIntegral', 'lineEdit_ZPosDerivative',
                          'lineEdit_XPosProportional', 'lineEdit_XPosIntegral', 'lineEdit_XPosDerivative']
        validator = QDoubleValidator(self)
        validator.setRange(0, 10)
        validator.setNotation(QDoubleValidator.StandardNotation)
        validator.setDecimals(2)
        # set validator in one line
        # [eval('self.' + name + '.setValidator(validator)') for name in lineEdit_names]  # error
        symbols = {"self": self, 'validator': validator}
        [eval('self' + '.' + name + '.setValidator(validator)', symbols) for name in lineEdit_names]

    def get_tello_status(self):
        return self.tello_status_tracker.status

    def register_callback(self):
        # self.machine_signal.connect(self.machine_cb)
        self.stream_processor.stream_signal.connect(self.update_processed_info_cb)
        self.visualizer.visualization_signal.connect(self.speech_cb)
        # self.position_controller.output_signal.connect()
        # self.tello_status_tracker.status_signal.connect(self.handle_status_cb)

        # pushButton group
        # running
        self.pushButton_start.clicked.connect(self.start_cb)
        self.pushButton_land.clicked.connect(self.land_cb)
        # PID parameters setting
        self.pushButton_setYawParameters.clicked.connect(self.setPID_cb)
        self.pushButton_setZPosParameters.clicked.connect(self.setPID_cb)
        self.pushButton_setXPosParameters.clicked.connect(self.setPID_cb)

        # radioButton group
        self.radioButton_manual.toggled.connect(self.ratio_manual_cb)
        self.radioButton_auto.toggled.connect(self.ratio_auto_cb)

    def ratio_manual_cb(self):
        self.pushButton_up.setEnabled(True)
        self.pushButton_down.setEnabled(True)
        self.pushButton_turnLeft.setEnabled(True)
        self.pushButton_turnRight.setEnabled(True)
        self.pushButton_forward.setEnabled(True)
        self.pushButton_backward.setEnabled(True)

        self.position_controller.auto_mode = False

    def ratio_auto_cb(self):
        self.pushButton_up.setDisabled(True)
        self.pushButton_down.setDisabled(True)
        self.pushButton_turnLeft.setDisabled(True)
        self.pushButton_turnRight.setDisabled(True)
        self.pushButton_forward.setDisabled(True)
        self.pushButton_backward.setDisabled(True)

        self.position_controller.auto_mode = True

    def setPID_cb(self):
        pid_type = ORDERS[self.tabWidget.currentIndex()]
        # list elements order: P -> I -> D
        line_edits_group = [f'lineEdit_{pid_type.capitalize()}Proportional',
                            f'lineEdit_{pid_type.capitalize()}Integral',
                            f'lineEdit_{pid_type.capitalize()}Derivative']
        set_vals = tuple(map(lambda le: float(le.text()), line_edits_group))
        self.position_controller.tunings(pid_type, values=set_vals)
        logging.info(f"PID setting on {pid_type} controller: {set_vals}")

    def handle_status_cb(self, msg):
        # if low-battery, warn and land
        if self.tello_status_tracker.status['battery'] <= 10:
            msg = "Low Battery Warning. Force Landing!"
            self.tello.land()
        else:
            pass

        self.statusbar.setStyleSheet("color: red")
        self.statusbar.showMessage(msg)

    def compute_measurement(self, frame, detect_ret) -> list:
        """
        :param frame:
        :param infer_ret:
        :return: [nose's pixel-val on width-dir, nose's pixel-val on height-dir, head's width pixel-val] yaw->zpos->xpox
        """
        image_h, image_w = frame.shape[:2]
        frame_landmarks = detect_ret.landmark
        # points_list = [[int(landmark.x * image_w), int(landmark.y * image_h)] for landmark in frame_landmarks]  # 33点

        # for item that not to control, return measurement of 'None'
        yaw_meas = frame_landmarks[0].x * image_w
        zpos_meas = frame_landmarks[0].y * image_h

        # ret_faces = self.face_detector.inference(frame)
        # xpos_meas = ret_faces[0][2] if ret_faces is not None else None
        xpos_meas = None

        measurement = [yaw_meas, zpos_meas, xpos_meas]  # orders: yaw->zpos->xpos

        return measurement

    def update_processed_info_cb(self, stream_signal):
        """Postprocessing of ProcessStreamThread's each iteration"""
        frame, detect_ret = stream_signal

        if self.position_controller.auto_mode:
            measurement = self.compute_measurement(frame, detect_ret)
            self.controller_input_queue.append(measurement)

    def speech_cb(self, infer_signal):
        # update tricks
        if not self.is_people:
            if infer_signal['landmark'] is not None:
                self.is_people = True
                msg = "检测到有行人出现"
                logging.info(msg)
                self.speech_msg_queue.append(msg)
        elif infer_signal['landmark'] is None:
            self.is_people = False
            self.is_ikun = False

        if not self.is_ikun:
            if infer_signal['is_ikun']:
                self.is_ikun = True
                msg = "发现真IKUN"
                logging.info(msg)
                self.speech_msg_queue.append(msg)

    def start_cb(self):
        logging.info("start-button pushed")
        # start a stream-processing thread
        self.stream_processor.start()
        self.visualizer.start()
        self.tello_status_tracker.start()

        while not self.stream_processor.isStreaming:
            logging.info("Stream not available")
            time.sleep(1)
        logging.info("Streaming on!")

        frame_h, frame_w = self.stream_processor.get_frame_shape()

        # config and start a position-control thread.
        # eyes locate at 1/2-width, 3/4-height of the image.
        # The width of the head needs to occupy 1/10-width of the image.
        # pid_setpoints = {'yaw': int(frame_w / 2), 'xpos': int(frame_w / 10), 'zpos': int(frame_h * 3 / 4)}
        # self.position_controller.set_controller_setpoints(pid_setpoints)
        # self.position_controller.start()

        # take-off then leave this func
        tello_battery = self.tello.get_battery()
        if tello_battery > 30:
            logging.info(f"Sufficient battery power: {tello_battery}%. TelloDrone takeoff!")
            self.tello.takeoff()
            self.tello.move_up(50)
        else:
            logging.error(f"TelloDrone low battery: {tello_battery}%. Please charge first!")

    def land_cb(self):
        logging.info("stop-button pushed")
        self.tello.land()


if __name__ == '__main__':
    RECOGNIZER_PATH = './research/saved_model/ikun_classifier45'
    YUNET_PATH = './research/yunet_model/face_detection_yunet_2022mar.onnx'
    MPOSE_CONFIG = {'model_complexity': 2,
                    'enable_segmentation': False,
                    'min_detection_confidence': 0.5,
                    'min_tracking_confidence': 0.5}

    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow(RECOGNIZER_PATH, YUNET_PATH, MPOSE_CONFIG)
    window.show()
    sys.exit(app.exec_())
