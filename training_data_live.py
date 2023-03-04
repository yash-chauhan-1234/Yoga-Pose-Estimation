import copy
import csv
import itertools
import os
import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

image_out='coordinate_imgs'
keypoints={1:0, 2:0, 3:0, 4:0, 5:0, 6:0}

def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
    if key == 110:  # n
        mode = 0
    if key == 107:  # k
        mode = 1
    if key == 104:  # h
        mode = 2
    return number, mode

def logging_csv(number, mode, landmark_list):
    if mode == 0:
        pass
    if mode == 1 and (0 <= number <= 9):
        csv_path = 'keypoint.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
    return

def logging_img(number, mode, image, pose_landmarks):
    if mode==0:
        pass
    if mode==1 and (0<=number<=9):
        output_frame = image.copy()
        mp_drawing.draw_landmarks(
            image=output_frame,
            landmark_list=pose_landmarks,
            connections=mp_pose.POSE_CONNECTIONS)
        # Flip the image horizontally for a selfie-view display.
        output_frame = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)
        
        try: 
            # print(number, mode)
            # print(os.path.join(image_out, f"Class{number}_{keypoints[number]}"))
            cv2.imwrite(os.path.join('coordinate_imgs', f"Class{number}_{keypoints[number]}.png"), output_frame)
        # print(keypoints[number]+1)
            keypoints[number]+=1
        except:
        #     print("except")
            pass

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list

mode=0
i=0


# For webcam input:
cap = cv2.VideoCapture(0)
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:

  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    # print(results.pose_landmarks)
    # Draw the pose annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    k=cv2.waitKey(50)
    
    if k==27:
        break
    
    number, mode = select_mode(k, mode)
    # print(number, mode)

    pose_landmarks=results.pose_landmarks
    if pose_landmarks is not None and number>0:

        logging_img(number, mode, image, pose_landmarks)
        # assert len(pose_landmarks.landmark) == 33, 'Unexpected number of predicted pose landmarks: {}'.format(len(pose_landmarks.landmark))
        frame_height, frame_width = image.copy().shape[:2]
        landmark_point=[]
        for lmk in pose_landmarks.landmark:
            landmark_x=min(int(lmk.x*frame_width), frame_width-1)
            landmark_y=min(int(lmk.y*frame_height), frame_height-1)

            landmark_point.append([landmark_x, landmark_y])

        pre_processed_landmark_list=pre_process_landmark(landmark_point)

        # Map pose landmarks from [0, 1] range to absolute coordinates to get
        # correct aspect ratio.
        # pose_landmarks *= np.array([frame_width, frame_height])

        # Write pose sample to CSV.
        # pose_landmarks = np.around(pose_landmarks, 5).flatten().astype(np.str).tolist()
        logging_csv(number, mode, pre_processed_landmark_list)
        
        

    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    # Flip the image horizontally for a selfie-view display.
    # output_frame = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)
    # cv2.imwrite(os.path.join('coordinate_imgs', f""), output_frame)


    cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
    
cap.release()

print(keypoints)