import cv2
import numpy as np
import os
import pickle
from tqdm import tqdm

big_image_shape = (1920, 1080)
sub_image_shape = (480, 270)
videos_num_width = int(big_image_shape[0] / sub_image_shape[0])
videos_num_height = int(big_image_shape[1] / sub_image_shape[1])
videos_num = videos_num_height * videos_num_width
print(f"need videos samples number: {videos_num}")
videos_num = int(videos_num)


if __name__ == '__main__':
    video_root = 'E:\\Video\\TelloDroneDetection\\原始视频素材\\negative'

    cache_path = 'all_video_frames.pkl'
    if not os.path.exists(cache_path):
        with open(cache_path, 'wb') as f:
            all_video_frames = []
            min_frames_num = 1024 ** 2
            for filename in os.listdir(video_root):
                vid_path = os.path.join(video_root, filename)

                cap = cv2.VideoCapture(vid_path)
                if not cap.isOpened():
                    print("Error Opening video File")
                    raise IOError

                frames = []
                while cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        frames.append(frame)
                    else:
                        break
                all_video_frames.append(frames)
                min_frames_num = min(min_frames_num, len(frames))

                cap.release()
                cv2.destroyAllWindows()
            print(f'min_frames_num: {min_frames_num}')

            all_video_frames = np.array([frames[:min_frames_num] for frames in all_video_frames])
            pickle.dump(all_video_frames, f)
            print("successfuly cached.")
    else:
        with open(cache_path, 'rb') as f:
            all_video_frames = pickle.load(f)
            print("successfuly loaded.")

    out = cv2.VideoWriter('negative_patch.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 25, big_image_shape)
    for frame_cnt in tqdm(range(all_video_frames.shape[1])):
        frame = np.zeros((big_image_shape[1], big_image_shape[0], 3)).astype(np.uint8)
        i_w, j_h = 0, 0
        for i in range(videos_num_width):
            for j in range(videos_num_height):
                frame[j*sub_image_shape[1]: ((j+1)*sub_image_shape[1]), i*(sub_image_shape[0]): ((i+1)*sub_image_shape[0])] = \
                    cv2.resize(all_video_frames[i * videos_num_width + j][frame_cnt], sub_image_shape)

        #cv2.imshow('negative', cv2.resize(frame, (1400, 788)))
        cv2.waitKey(30)

        out.write(frame)
    out.release()


