"""https://stackoverflow.com/questions/13901997/kalman-2d-filter-in-python"""
import traceback

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import moviepy.editor as mpy
import mediapipe as mp
import numpy as np
import mimetypes
import os
import cv2
import pickle
from Kalman2D import kalman_xy
from PoseDefinition import PoseDefinition


class TrajectoryPlot:
    def __init__(self, model_complexity, enable_segmentation, min_detection_confidence, min_tracking_confidence):
        self.pose = mp.solutions.pose.Pose(
            model_complexity=model_complexity,
            enable_segmentation=enable_segmentation,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.it_id = 0

    @staticmethod
    def extract_video_frames(vid_path, sample_stride=0):
        """
        @param sample_stride: 'stride=0' will reserve all frames
        """
        cap = cv2.VideoCapture(vid_path)
        if not cap.isOpened():
            print("Error Opening video File")
            raise IOError

        sample_frames = []
        frames_cnt = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                if not frames_cnt % sample_stride:
                    sample_frames.append(frame)
            else:
                break

            frames_cnt += 1

        cap.release()
        cv2.destroyAllWindows()

        return sample_frames

    def plot_pose(self, ax, pose_landmarks):
        # plot key-points
        for index, point in enumerate(pose_landmarks):
            ax.plot(point[0], point[1], marker="o", markersize=4, markeredgecolor="red", markerfacecolor="green")

        # plot edges
        for id_link in PoseDefinition.ALL_ID_LINKs:
            # ax.plot(pose_landmarks[id_link, 0], pose_landmarks[id_link, 1], color='#900302', marker='+', linestyle='-')
            ax.plot(pose_landmarks[id_link, 0], pose_landmarks[id_link, 1], color='#900302')

    def cv2_draw_pose(self, img, pts):
        for edge in PoseDefinition.EDGES:
            cv2.line(img, tuple(pts[edge[0]]), tuple(pts[edge[1]]), (0, 255, 0), 5)

    def frame_inference(self, img):
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image)
        return results.pose_landmarks

    def detect_key_point(self, frames):
        landmarks = []  # shape of (frames-num, key-points-num, 2)
        image_h, image_w = frames[0].shape[:2]
        for frame in frames:
            pose_landmarks = self.frame_inference(frame)
            if pose_landmarks is not None:
                frame_landmarks = pose_landmarks.landmark
                points_list = [[int(landmark.x * image_w), int(landmark.y * image_h)] for landmark in frame_landmarks]
                landmarks.append(points_list)
            else:
                landmarks.append(None)

        landmarks_valid = [landmark for landmark in landmarks if landmark is not None]
        landmarks_valid = np.array(landmarks_valid)
        landmarks = np.array(landmarks)

        return landmarks, landmarks_valid

    def filter_2d(self, meas_points):
        filtered_points = []
        s = np.matrix('0. 0. 0. 0.').T
        P = np.matrix(np.eye(4)) * 1000  # initial uncertainty
        R = 0.01 ** 2
        for meas in meas_points:
            s, P = kalman_xy(s, P, meas, R)
            filtered_points.append((s[:2]).tolist())

        filtered_points = np.array(filtered_points).squeeze()

        return filtered_points

    def plot_traj(self, vid_path, key_point_ids: list, mode=0, stride=0):
        """绘制所有轨迹点"""
        # step1: extract frames
        frames = self.extract_video_frames(vid_path, sample_stride=stride)
        print(f"Extracted frames numbers: {len(frames)}")

        frame_h, frame_w = frames[0].shape[:2]

        # step2: detect key-point from each frame (save only one person's landmarks)
        cache_path = 'static_plot_data.pkl'
        if not os.path.exists(cache_path):
            landmarks, landmarks_valid = self.detect_key_point(frames)
            with open(cache_path, 'wb') as f:
                pickle.dump((landmarks, landmarks_valid), f)
        else:
            with open(cache_path, 'rb') as f:
                landmarks, landmarks_valid = pickle.load(f)

        # step3: kalman-filter on single landmark
        meas_points = landmarks_valid[:, key_point_ids]
        meas_points = meas_points.mean(axis=1)
        filtered_points = self.filter_2d(meas_points)

        # step4: animate curve, full-pose and select-key-point using matplotlib
        # coordinate transformation (invert vertical direction)
        #landmarks_valid[..., 1] = frame_h - landmarks_valid[..., 1]
        landmarks_valid_vert = np.concatenate((landmarks_valid[..., [0]], frame_h - landmarks_valid[..., [1]]), axis=2)

        print(np.shape(filtered_points))
        filtered_points_vert = np.concatenate((filtered_points[:, [0]], frame_h - filtered_points[:, [1]]), axis=1)

        # set up the figure, the axis, and the plot element we want to animate
        fig = plt.figure(figsize=(8, 8))
        ax = plt.axes(xlim=(0, frame_w), ylim=(0, frame_h))  # -300))

        def setup():
            ax.clear()

            ax.set_aspect("equal")
            ax.set_xlim(0, frame_w)
            ax.set_ylim(0, frame_h)  # -300)

            ax.set_xlabel('x (px)')
            ax.set_ylabel('y (px)')

        points_iterator = iter(filtered_points_vert)

        # animation function.  This is called sequentially
        def animate_pose(i):
            setup()

            filtered_point = next(points_iterator)
            x = filtered_point[0]
            y = filtered_point[1]
            # print(x, y)
            ax.plot(x, y, 'bo', markersize=8)

            # print(landmarks_valid[self.it_id])
            self.plot_pose(ax, landmarks_valid_vert[self.it_id])

            self.it_id += 1
            if self.it_id == len(filtered_points_vert) - 1:
                anim.event_source.stop()

        def animate_fixed_nums_traj(i):
            setup()
            try:
                plotting_points.append(next(points_iterator))
                if len(plotting_points) > points_num:
                    plotting_points.pop(0)

                ax.plot([ele[0] for ele in plotting_points], [ele[1] for ele in plotting_points], 'bo', markersize=3)
            except IndexError as e:
                print("Index out of bounds")
                anim.event_source.stop()

            self.plot_pose(ax, landmarks_valid_vert[self.it_id])

            self.it_id += 1
            if self.it_id == len(filtered_points_vert) - 1:
                anim.event_source.stop()

        if mode == 0:
            animate = animate_pose
        else:
            plotting_points = []
            points_num = 30
            animate = animate_fixed_nums_traj

        # call the animator.  blit=True means only re-draw the parts that have changed.
        anim = animation.FuncAnimation(fig, animate, frames=len(filtered_points_vert), interval=1)
        # anim = animation.FuncAnimation(fig, animate, interval=20)
        # anim.save('animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
        plt.show()

        # step5: plot curve on origin image
        image_h, image_w = frames[0].shape[:2]
        res_h, res_w = (int(image_w/2), int(image_h/2))
        out = cv2.VideoWriter('static_plot.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (image_w, image_h))
        # Create solid alpha layer, same height and width as "img", filled with 128s
        alpha = np.full_like(frames[0][..., 0], 128)
        for index, frame in enumerate(frames):
            if landmarks[index] is not None:
                # Merge new alpha layer onto image with OpenCV "merge()"
                # semi_trans_frame = cv2.merge((frame, alpha))
                semi_trans_frame = frame

                # Calculate
                circle_center = tuple(filtered_points[index].astype(np.int))

                # plot lines
                pts = landmarks_valid[index]
                # cv2.polylines(semi_trans_frame, [pts], True, (0, 255, 255))
                self.cv2_draw_pose(semi_trans_frame, pts)

                # plot key-points
                cv2.circle(semi_trans_frame, circle_center, 10, (255, 120, 50), thickness=-1)

                out.write(semi_trans_frame)
                cv2.imshow('frame', cv2.resize(semi_trans_frame, (int(res_h*1.8), int(res_w*1.8))))

                # Press Q on keyboard to stop recording
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
        out.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    video_path = 'tieshankao43.mp4'

    traj_plotter = TrajectoryPlot(model_complexity=2,
                                  enable_segmentation=False,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)

    traj_plotter.plot_traj(vid_path=video_path, key_point_ids=[11, 12], mode=0, stride=1)
