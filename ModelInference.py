import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import Sequential
import mediapipe as mp
import numpy as np
import cv2
from itertools import chain

CLASS_NAMES = ['xiaoheizi', 'ikun']
SEQ_LENGTH = 45
COMPLEX_MODE = True
CLASS_NUM = 2
BATCH_SIZE = 4
INPUT_SHAPE = (SEQ_LENGTH, 2) if not COMPLEX_MODE else (SEQ_LENGTH, 33*2)


class PoseDetector:
    def __init__(self, model_complexity, enable_segmentation, min_detection_confidence, min_tracking_confidence):
        self.pose = mp.solutions.pose.Pose(model_complexity=model_complexity,
                                           enable_segmentation=enable_segmentation,
                                           min_detection_confidence=min_detection_confidence,
                                           min_tracking_confidence=min_tracking_confidence)

    def inference(self, frame_rgb):
        # image = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        ret = self.pose.process(frame_rgb)
        return ret.pose_landmarks


class IkunRecognizer:
    SEQUENCE_NUM = SEQ_LENGTH

    def __init__(self, model_path):
        self.model = self.load_model(model_path)  # tf.keras.models.load_model(saved_model_path)
        self.make_first_inference()

    def load_model(self, model_path, saved_model=False):
        if saved_model:
            return tf.keras.models.load_model(model_path)
        else:
            # step1: build graph
            model = Sequential([
                layers.Input(shape=INPUT_SHAPE),
                layers.GRU(32, return_sequences=False),
                layers.Dense(len(CLASS_NAMES), activation="softmax")
            ])
            model.compile(
                optimizer=optimizers.Adam(0.001),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"]
            )

            # step2: restore parameters
            model.load_weights(model_path)
            return model

    def make_first_inference(self):
        random_input = np.random.random_sample((1, self.SEQUENCE_NUM, 33*2))
        self.model.predict(random_input, verbose=0)

    def inference(self, mpose_output_list):
        landmarks = [[[landmark.x, landmark.y] for landmark in item.landmark] for item in mpose_output_list]
        model_input = np.expand_dims([list(chain.from_iterable(item)) for item in landmarks], 0)
        ret = self.model.predict(model_input, verbose=0)  # 0=silent, 1=progress-bar, 2=one-line-per-epoch

        return ret


class FaceDetector:
    def __init__(self, yunet_model_path, set_shape=(320, 320)):
        self.model = cv2.FaceDetectorYN.create(
            model=yunet_model_path,
            config='',
            input_size=(320, 320),
            score_threshold=0.9,
            nms_threshold=0.3,
            top_k=5000,
            backend_id=cv2.dnn.DNN_BACKEND_DEFAULT,
            target_id=cv2.dnn.DNN_TARGET_CPU
        )

        self.set_shape = set_shape
        self.model.setInputSize(set_shape)

    def inference(self, img):
        if img.shape[:2] != self.set_shape:
            self.model.setInputSize(img.shape[:2][::-1])  # set size should be (width, height) order
            self.set_shape = img.shape[:2]
        _, faces = self.model.detect(img)

        return faces
