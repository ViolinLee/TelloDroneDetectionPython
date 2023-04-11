import cv2
import numpy as np
from PoseDefinition import PoseDefinition
from PIL import Image, ImageDraw, ImageFont


class DataPipeline:
    def __init__(self):
        pass

    @staticmethod
    def put_chinese_text(img, text, org, color=(0, 255, 0), size=30):
        if isinstance(img, np.ndarray):
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        draw = ImageDraw.Draw(img)
        font_style = ImageFont.truetype("simsun.ttc", size, encoding="utf-8")
        draw.text(org, text, color, font=font_style)

        res = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)  # in-replace
        return res

    @staticmethod
    def draw_info(img, info: dict):
        """
        draw information at right-corner of the input image
        :param img:
        :param info: {'fps': 10, 'battery': 85, 'result': ''}
        :return:
        """
        corner_org = [img.shape[1] - 200, 50]
        y_step = 30

        for key, value in info.items():
            cv2.putText(img, f'{key}: {value}', corner_org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            corner_org[1] = corner_org[1] + y_step

        return img

    @staticmethod
    def pipe_pre(frame):
        # frame = cv2.flip(frame, 1)
        #frame = cv2.resize(frame, (640, 480))  # 720p(960,720)->480p(640,480)
        frame = cv2.resize(frame, (720, 540))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        return frame

    @staticmethod
    def pipe_post(frame: np.array, infer: dict, tello_status: dict):
        """
        :param frame:
        :param infer: {'landmark': [], 'is_ikun': boolean}
        :param tello_status: {'temperature': float, 'battery': float, 'flight_time': float}
        :return: processed-frame and skeleton_img
        """

        frame_height, frame_width = frame.shape[:2]

        # step1:
        frame = DataPipeline.draw_info(frame, tello_status)
        if infer['is_ikun'] is True:
            frame = DataPipeline.put_chinese_text(frame, "真IKUN", (100, 100))

        # step2:
        skeleton_img = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)  # malloc mem

        if infer['landmark']:
            frame_landmarks = infer['landmark'].landmark  # mediapipe's original processed output
            points_list = [[int(landmark.x * frame_width), int(landmark.y * frame_height)] for landmark in frame_landmarks]

            for edge in PoseDefinition.EDGES:
                cv2.line(skeleton_img, tuple(points_list[edge[0]]), tuple(points_list[edge[1]]), (0, 255, 0), 2)

        # center crop
        crop_width = frame_height * 360 / 540
        crop_start = int((frame_width - crop_width) / 2)
        skeleton_img_crop = skeleton_img[:, crop_start: frame_width - crop_start]
        skeleton_img_resize = cv2.resize(skeleton_img_crop, (360, 540))

        return frame, skeleton_img_resize
