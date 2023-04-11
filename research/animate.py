import mediapipe as mp
import cv2
import sys
import numpy as np
import tensorflow as tf
from itertools import chain
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import Sequential
from PIL import Image, ImageDraw, ImageFont


CLASS_NAMES = ['xiaoheizi', 'ikun']
SEQ_LENGTH = 45
COMPLEX_MODE = True
CLASS_NUM = 2
BATCH_SIZE = 4
INPUT_SHAPE = (SEQ_LENGTH, 2) if not COMPLEX_MODE else (SEQ_LENGTH, 33*2)
# Mediapipe model config
MODEL_COMPLEXITY = 2
ENABLE_SEGMENTATION = False
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5
# image-show window size configuration
MAX_WINDOW_WIDTH = 1920/3
MAX_WINDOW_HEIGHT = 1080/3
MAX_SIZE = np.array([MAX_WINDOW_WIDTH, MAX_WINDOW_HEIGHT]).astype(int)
SMALL_SIZE = (MAX_SIZE / 2).astype(int)


class Inference:
    def __init__(self, pose, cpt_path):
        self.pose = pose
        self.predictor = self.load_model(cpt_path)
        self.make_first_inference()

    def make_first_inference(self):
        random_input = np.random.random_sample((1, SEQ_LENGTH, 33*2))
        self.predictor.predict(random_input, verbose=1)  # mode. 0 = silent, 1 = progress bar, 2 = one line per epoch

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

    def detect_landmark_fp(self, frame):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.pose.process(image)
        return result

    def extract_entire_feature(self, frames):
        landmarks = [self.detect_landmark_fp(frame).pose_landmarks for frame in frames]
        valid_landmarks = [[[landmark.x, landmark.y] for landmark in item.landmark] for item in landmarks if
                           item is not None]  # (N, 33, 2)
        valid_samples = [list(chain.from_iterable(item)) for item in valid_landmarks][:SEQ_LENGTH]
        return valid_samples

    def pipeline(self, frames):
        assert len(frames) >= SEQ_LENGTH, f'length of frames sequence is smaller than {SEQ_LENGTH}.'

        features_seq = self.extract_entire_feature(frames)
        model_input = np.array(features_seq)

        if model_input.shape[0] == SEQ_LENGTH:

            ret = self.predictor.predict(model_input)
        else:
            ret = None
        return ret

    def pipeline_features(self, landmark_features):
        landmarks = [[[landmark.x, landmark.y] for landmark in item.landmark] for item in landmark_features]
        model_input = np.expand_dims([list(chain.from_iterable(item)) for item in landmarks], 0)
        ret = self.predictor.predict(model_input, verbose=0)
        return ret


def put_chinese_text(img, text, org, color=(255, 100, 100), size=35):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    draw = ImageDraw.Draw(img)
    font_style = ImageFont.truetype("simsun.ttc", size, encoding="utf-8")
    draw.text(org, text, color, font=font_style)

    res = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)  # in-replace
    return res


def animate(vid_path, cpt_path, mpose):
    # step0: create model instance
    engine = Inference(mpose, cpt_path)

    # step1: extract original frames and video info (shape, fps)
    cap = cv2.VideoCapture(vid_path)
    if not cap.isOpened():
        print("Error Opening video File")
        raise IOError

    # get fps
    (major_ver, minor_ver, subminor_ver) = cv2.__version__.split('.')
    if int(major_ver) < 3:
        fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
    else:
        fps = cap.get(cv2.CAP_PROP_FPS)
    cycle_ms = 1  # int(1000/fps)

    # get frame shape
    frame_size = np.array([cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)])
    resize_size = (frame_size * (MAX_SIZE/frame_size).min()).astype(int)
    resize_size_small = (resize_size / 2).astype(int)
    # print(resize_size)

    # step2: read and inference
    frames_features = []
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, resize_size)

            frame_landmarks = engine.detect_landmark_fp(cv2.resize(frame, resize_size_small)).pose_landmarks
            frames_features.append(frame_landmarks) if frame_landmarks is not None else None
            frames_features.pop(0) if len(frames_features) > SEQ_LENGTH else None

            # ret = engine.pipeline(frames) if len(frames) == SEQ_LENGTH else None
            ret = engine.pipeline_features(frames_features) if len(frames_features) == SEQ_LENGTH else None
            print(ret[0]) if ret is not None else None
            if ret is not None and CLASS_NAMES[np.argmax(ret)] == 'ikun':
                print('The True IKun!')
                frame = put_chinese_text(frame, "鉴定为：真IKun", (30, 30), color=(255, 50, 50), size=35)
            else:
                pass

            cv2.imshow("Frame", cv2.resize(frame, (int(frame.shape[1]*2.2), int(frame.shape[0]*2.2))))
            if cv2.waitKey(cycle_ms) & 0xFF == ord("q"):
                break
        else:
            break
    cap.release()


if __name__ == '__main__':
    # configuration
    video_path = 'yuanban_short.mp4'
    checkpoint_path = 'saved_model/ikun_classifier45'

    pose = mp.solutions.pose.Pose(model_complexity=MODEL_COMPLEXITY,
                                  enable_segmentation=ENABLE_SEGMENTATION,
                                  min_detection_confidence=MIN_DETECTION_CONFIDENCE,
                                  min_tracking_confidence=MIN_TRACKING_CONFIDENCE)

    # running
    animate(video_path, checkpoint_path, pose)
