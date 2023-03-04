import copy
import itertools
import cv2 as cv
import numpy as np
import mediapipe as mp
import tensorflow as tf

# from MODEL import yoga

def main():
    cap = cv.VideoCapture(0)
    # cap.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
    # cap.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)

    mp_pose = mp.solutions.pose
    pose=mp_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    # model=yoga()

    posess=["Cat Pose", "Downward Facing Dog Pose", "Side Lunge", "Lunge", "Pashchimotanasana", "Butterfly Pose"]
    
    mode=0

    while True:
        key=cv.waitKey(10)

        if key==27:
            break

        number, mode=select_mode(key, mode)

        success, image=cap.read()
        if not success:
            break

        image=cv.flip(image, 1)
        debug_img=image.copy()

        # image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        # print(np.array(image).shape)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True

        if results.pose_landmarks is not None:
            pose_landmarks=results.pose_landmarks
        # assert len(pose_landmarks.landmark) == 33, 'Unexpected number of predicted pose landmarks: {}'.format(len(pose_landmarks.landmark))
            frame_height, frame_width = image.copy().shape[:2]
            landmark_point=[]
            for lmk in pose_landmarks.landmark:
                landmark_x=min(int(lmk.x*frame_width), frame_width-1)
                landmark_y=min(int(lmk.y*frame_height), frame_height-1)

                landmark_point.append([landmark_x, landmark_y])

            pre_processed_landmark_list=pre_process_landmark(landmark_point)

            if mode==2:
                pose_id=model_run(np.array(pre_processed_landmark_list))

                if pose_id is not None:
                    print(posess[pose_id])
                    image=draw_info_text(image, posess[pose_id])


            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

            

        cv.imshow('Pose Recognition', image)

    cap.release()
    # cv.destroyAllWindows()


def model_run(data):
    MODEL_PATH='D:\College\sem 5\ML\Project\MODEL\yoga\model.tflite'
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], np.array([data], dtype='float32'))

    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    return np.argmax(np.squeeze(output_data))

def img_data(img):
    image=cv.resize(img, (256, 256))
    return np.array(image)


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

def draw_info_text(image, text):
    cv.putText(image, text, (50, 50), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

    return image

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


if __name__=='__main__':
    main()