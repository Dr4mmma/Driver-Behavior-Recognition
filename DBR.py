import platform
import cv2
import numpy as np
import time
import socket
import KNNclassifier
import pickle
import SEQclassifier
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import threading
import torch.hub
import os
import model
from head_pose_estimation.pose_estimator import PoseEstimator
from head_pose_estimation.stabilizer import Stabilizer
from head_pose_estimation.visualization import *
from head_pose_estimation.misc import *
from PIL import Image
from torchvision import transforms
from grad_cam import BackPropagation
from os.path import dirname, join
from playsound import playsound
from argparse import ArgumentParser
from multiprocessing import Process, Queue


timebasedrow= time.time()
timebasedis= time.time()
timerundrow= time.time()
timerundis= time.time()

face_cascade = cv2.CascadeClassifier(r'C:\Users\moham\haar_models\haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(r'C:\Users\moham\haar_models\haarcascade_eye.xml')
MyModel="BlinkModel.t7"
mp3File = 'alarm.mp3'

shape = (24,24)
classes = [
    'Close',
    'Open',
]

eyess=[]
cface=0
causse =[]
seqclass = []
alarm = []
alarmnot = []

def imp():
    try:
        return causse.pop()
    except:
        return 0


def cause(value):
    causse.append(value)


def preprocess(image_path):
    global cface
    transform_test = transforms.Compose([
        transforms.ToTensor()
    ])
    image = cv2.imread(image_path[r'C:\Users\moham'])
    faces = face_cascade.detectMultiScale(
        image,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(1, 1),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    if len(faces) == 0:
        ...
    else:
        cface = 1
        (x, y, w, h) = faces[0]
        face = image[y:y + h, x:x + w]
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 0), 2)
        roi_color = image[y:y + h, x:x + w]
        """
        Depending on the quality of your camera, this number can vary 
        between 10 and 40, since this is the "sensitivity" to detect the eyes.
        """
        sensi = 20
        eyes = eye_cascade.detectMultiScale(face, 1.3, sensi)
        i = 0
        for (ex, ey, ew, eh) in eyes:
            (x, y, w, h) = eyes[i]
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            eye = face[y:y + h, x:x + w]
            eye = cv2.resize(eye, shape)
            eyess.append([transform_test(Image.fromarray(eye).convert('L')), eye, cv2.resize(face, (48, 48))])
            i = i + 1
    cv2.imwrite(r'C:\Users\moham\temp-images\display.jpg', image)


def eye_status(image, name, net):
    img = torch.stack([image[name]])
    bp = BackPropagation(model=net)
    probs, ids = bp.forward(img)
    actual_status = ids[:, 0]
    prob = probs.data[:, 0]
    if actual_status == 0:
        prob = probs.data[:, 1]

    # print(name,classes[actual_status.data], probs.data[:,0] * 100)
    return classes[actual_status.data]


def func(imag, modl):
    drow(images=[{r'C:\Users\moham': imag, 'eye': (0, 0, 0, 0)}], model_name=modl)


def drow(images, model_name):
    global eyess
    global cface
    global timebasedrow
    global timebasedis
    global timerundrow
    global timerundis
    net = model.Model(num_classes=len(classes))
    checkpoint = torch.load(os.path.join(r'C:\Users\moham\Model', model_name), map_location=torch.device('cpu'))
    net.load_state_dict(checkpoint['net'])
    net.eval()

    flag = 1
    status = ""
    for i, image in enumerate(images):
        if (flag):
            preprocess(image)
            flag = 0
        if cface == 0:
            image = cv2.imread(r"C:\Users\moham\temp-images/display.jpg")
            image = cv2.putText(image, 'No face Detected', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,
                                cv2.LINE_AA)
            cv2.imwrite(r'C:\Users\moham\temp-images\display.jpg', image)
            timebasedrow = time.time()
            timebasedis = time.time()
            timerundrow = time.time()
            timerundis = time.time()
        elif (len(eyess) != 0):
            eye, eye_raw, face = eyess[i]
            image['eye'] = eye
            image['raw'] = eye_raw
            image['face'] = face
            timebasedrow = time.time()
            timerundrow = time.time()
            for index, image in enumerate(images):
                status = eye_status(image, 'eye', net)
                if (status == "Close"):
                    timerundis = time.time()
                    if ((timerundis - timebasedis) > 1.5):
                        image = cv2.imread(r'C:\Users\moham\temp-images\display.jpg')
                        image = cv2.putText(image, 'Distracted', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,
                                            cv2.LINE_AA)
                        cv2.imwrite(r'C:\Users\moham\temp-images\display.jpg', image)
                        cause1 = 'Distracted'
                        print(cause1)
                        cause(cause1)

        else:
            timerundrow = time.time()
            if ((timerundrow - timebasedrow) > 3):
                image = cv2.imread(r'C:\Users\moham\temp-images\display.jpg')
                image = cv2.putText(image, 'Drowsy', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.imwrite(r'C:\Users\moham\temp-images\display.jpg', image)
                cause2 = 'Drowsy'
                print(cause2)
                cause(cause2)



def get_face(detector, img_queue, box_queue, cpu=False):
    if cpu:
        while True:
            image = img_queue.get()
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            try:
                box = detector(image)[0]
                x1 = box.left()
                y1 = box.top()
                x2 = box.right()
                y2 = box.bottom()
                box_queue.put([x1, y1, x2, y2])
            except:
                box_queue.put(None)
    else:
        while True:
            image = img_queue.get()
            box = detector.extract_cnn_facebox(image)
            box_queue.put(box)

def running_on_jetson_nano():
    # To make the same code work on a laptop or on a Jetson Nano, we'll detect when we are running on the Nano
    # so that we can access the camera correctly in that case.
    # On a normal Intel laptop, platform.machine() will be "x86_64" instead of "aarch64"
    return platform.machine() == "aarch64"

def get_jetson_gstreamer_source(capture_width=1280, capture_height=720, display_width=1280, display_height=720, framerate=60, flip_method=0):
    """
    Return an OpenCV-compatible video source description that uses gstreamer to capture video from the camera on a Jetson Nano
    """
    return (
            f'nvarguscamerasrc ! video/x-raw(memory:NVMM), ' +
            f'width=(int){capture_width}, height=(int){capture_height}, ' +
            f'format=(string)NV12, framerate=(fraction){framerate}/1 ! ' +
            f'nvvidconv flip-method={flip_method} ! ' +
            f'video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! ' +
            'videoconvert ! video/x-raw, format=(string)BGR ! appsink'
            )

def main():
    # Setup face detection models
    if args.cpu: # use dlib to do face detection and facial landmark detection
        import dlib
        face_detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor('head_pose_estimation/assets/shape_predictor_68_face_landmarks.dat')
    else: # use better models on GPU
        import face_alignment
        face_detector = MarkDetector()
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
    if running_on_jetson_nano():
        # Accessing the camera with OpenCV on a Jetson Nano requires gstreamer with a custom gstreamer source string
        cap = cv2.VideoCapture(get_jetson_gstreamer_source(), cv2.CAP_GSTREAMER)
    else:
        # Accessing the camera with OpenCV on a laptop just requires passing in the number of the webcam (usually 0)
        # Note: You can pass in a filename instead if you want to process a video file instead of a live camera stream
        cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    _, sample_frame = cap.read()

    # Setup process and queues for multiprocessing.
    img_queue = Queue()
    box_queue = Queue()
    img_queue.put(sample_frame)
    box_process = Process(target=get_face, args=(
        face_detector, img_queue, box_queue, args.cpu,))
    box_process.start()

    # Introduce pose estimator to solve pose. Get one frame to setup the
    # estimator according to the image size.
    pose_estimator = PoseEstimator(img_size=sample_frame.shape[:2])

    # Introduce scalar stabilizers for pose.
    pose_stabilizers = [Stabilizer(
                        state_num=2,
                        measure_num=1,
                        cov_process=0.01,
                        cov_measure=0.1) for _ in range(8)]

    if args.connect:
        address = ('127.0.0.1', 5066)
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(address)

    ts = []
    frame_count = 0
    while True:
        _, frame = cap.read()
        frame = cv2.flip(frame, 2)
        frame_count += 1
        if args.connect and frame_count > 60:  # send information to unity
            msg = '%.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f' % \
                  (roll, pitch, yaw, min_ear, mar, mdst, steady_pose[6], steady_pose[7])
            s.send(bytes(msg, "utf-8"))

        t = time.time()
        ########################################added by me
        if frame_count > 150:
            msg = '%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n' % \
                  (roll, pitch, yaw, steady_pose[3], steady_pose[4], steady_pose[5], steady_pose[6], steady_pose[7])

            X=np.array([roll, pitch, yaw, steady_pose[3], steady_pose[4], steady_pose[5], steady_pose[6], steady_pose[7]])
            try:
                Pridected_GZ = KNNclassifier.KNN_model.predict(X.reshape(1, -1))
            except:
                Pridected_GZ='nothing'

            Prd_GZ = str(Pridected_GZ)
            Prd_GZ = Prd_GZ.replace("[", "")
            Prd_GZ = Prd_GZ.replace("]", "")
            Prd_GZ = Prd_GZ.replace("'", "")
            global eyess
            global cface
            eyess = []
            cface = 0
            ret, img = cap.read()
            cv2.imwrite(r'C:\Users\moham\temp-images\img.jpg', img)
            func(r'C:\Users\moham\temp-images\img.jpg', MyModel)
            if Prd_GZ == 'Front mirror':
                seqclass.append(0.0100)
            elif Prd_GZ == 'Left mirror':
                seqclass.append(0.0500)
            elif Prd_GZ == 'Right mirror':
                seqclass.append(0.0800)
            elif Prd_GZ == 'Rear mirror':
                seqclass.append(0.0250)
            elif Prd_GZ == 'Center Console':
                seqclass.append(0.0356)
            elif Prd_GZ == 'Dashboard':
                seqclass.append(0.1210)
            if len(seqclass) == 26:
                seq = '%.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f\n' % \
                  (seqclass[0], seqclass[1], seqclass[2], seqclass[3], seqclass[4], seqclass[5], seqclass[6], seqclass[7], seqclass[8], seqclass[9], seqclass[10], seqclass[11], seqclass[12], seqclass[13], seqclass[14], seqclass[15], seqclass[16], seqclass[17], seqclass[18], seqclass[19], seqclass[20], seqclass[21], seqclass[22], seqclass[23], seqclass[24], seqclass[25])
                Y = np.array(
                    [seqclass[0], seqclass[1], seqclass[2], seqclass[3], seqclass[4], seqclass[5], seqclass[6],
                     seqclass[7],
                     seqclass[8], seqclass[9], seqclass[10], seqclass[11], seqclass[12], seqclass[13], seqclass[14],
                     seqclass[15],
                     seqclass[16], seqclass[17], seqclass[18], seqclass[19], seqclass[20], seqclass[21], seqclass[22],
                     seqclass[23],
                     seqclass[24], seqclass[25]])
                Pridected_mode = SEQclassifier.KNN_model.predict(Y.reshape(1, -1))
                Prd_MD = str(Pridected_mode)
                Prd_MD = Prd_MD.replace("[", "")
                Prd_MD = Prd_MD.replace("]", "")
                Prd_MD = Prd_MD.replace("'", "")
                Prd_CS = str(imp())
                if Prd_MD == 'Apnormal' or Prd_CS == 'Drowsy' or Prd_CS == 'Distracted':
                    causse.clear()
                    cv2.putText(frame, 'Apnormal Driving Mode', (300, 30), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
                    alarm.append("Apnormal")
                else:
                    cv2.putText(frame, 'Normal Driving Mode', (300, 30), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
                    alarmnot.append("Normal")
                seqclass.clear()
            if len(alarm) == 2:
                playsound(mp3File)
                alarm.clear()
            if len(alarmnot) == 1:
                alarm.clear()
                alarmnot.clear()
            cv2.putText(frame, Prd_GZ, (400, 100), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)

        # Pose estimation by 3 steps:
        # 1. detect face;
        # 2. detect landmarks;
        # 3. estimate pose

        img_queue.put(frame)
        facebox = box_queue.get()

        if facebox is not None:
            # Do face detection, facial landmark detection and iris detection.
            if args.cpu:
                face = dlib.rectangle(left=facebox[0], top=facebox[1], right=facebox[2], bottom=facebox[3])
                marks = shape_to_np(predictor(frame, face))
            else:
                face_img = frame[facebox[1]: facebox[3], facebox[0]: facebox[2]]
                marks = fa.get_landmarks(face_img[:,:,::-1],
                        detected_faces=[(0, 0, facebox[2]-facebox[0], facebox[3]-facebox[1])])[-1]
                marks[:, 0] += facebox[0]
                marks[:, 1] += facebox[1]

            x_l, y_l, ll, lu = detect_iris(frame, marks, "left")
            x_r, y_r, rl, ru = detect_iris(frame, marks, "right")

            # Try pose estimation with 68 points.
            R, T = pose_estimator.solve_pose_by_68_points(marks)
            pose = list(R) + list(T)
            # Add iris positions to stabilize.
            pose+= [(ll+rl)/2.0, (lu+ru)/2.0]

            # Stabilize the pose.
            steady_pose = []
            pose_np = np.array(pose).flatten()
            for value, ps_stb in zip(pose_np, pose_stabilizers):
                ps_stb.update([value])
                steady_pose.append(ps_stb.state[0])

            if args.debug:

                # show iris.
                if x_l > 0 and y_l > 0:
                    draw_iris(frame, x_l, y_l)
                if x_r > 0 and y_r > 0:
                    draw_iris(frame, x_r, y_r)


                # show face landmarks.
                draw_marks(frame, marks, color=(0, 255, 0))

                # show facebox.
                draw_box(frame, [facebox])

                # draw stable pose annotation on frame.
                pose_estimator.draw_annotation_box(
                    frame, np.expand_dims(steady_pose[:3],0), np.expand_dims(steady_pose[3:6],0),
                    color=(128, 255, 128))

                # draw head axes on frame.
                pose_estimator.draw_axes(frame, np.expand_dims(steady_pose[:3],0),
                                         np.expand_dims(steady_pose[3:6],0))

            roll = np.clip(-(180+np.degrees(steady_pose[2])), -50, 50)
            pitch = np.clip(-(np.degrees(steady_pose[1]))-15, -40, 40)
            yaw = np.clip(-(np.degrees(steady_pose[0])), -50, 50)
            min_ear = min(eye_aspect_ratio(marks[36:42]), eye_aspect_ratio(marks[42:48]))
            mar = mouth_aspect_ration(marks[60:68])
            mdst = mouth_distance(marks[60:68])/(facebox[2]-facebox[0])

            if args.connect and frame_count > 60:
                msg = '%.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f'% \
                      (roll, pitch, yaw, min_ear, mar, mdst, steady_pose[6], steady_pose[7])
                s.send(bytes(msg, "utf-8"))



        dt = time.time() - t
        try:
            FPS = int(1/dt)
        except:
            dt1 = time.time()
            FPS = int(1/dt1)
        ts += [dt]
        print('\r', '%.3f'%dt, end=' ')

        if args.debug:
            draw_FPS(frame, FPS)
            cv2.imshow("face", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): # press q to exit.
                break

    # Clean up the multiprocessing process.
    box_process.terminate()
    box_process.join()
    cap.release()
    if args.connect:
        s.close()
    if args.debug:
        cv2.destroyAllWindows()
    print('%.3f'%np.mean(ts))

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--cam", type=int,
                        help="specify the camera number if you have multiple cameras",
                        default=0)
    parser.add_argument("--cpu", action="store_true",
                        help="use cpu to do face detection and facial landmark detection",
                        default=False)
    parser.add_argument("--debug", action="store_true",
                        help="show camera image to debug (need to uncomment to show results)",
                        default=False)
    parser.add_argument("--connect", action="store_true",
                        help="connect to unity character",
                        default=False)
    args = parser.parse_args()
    timebasedrow = time.time()
    timebasedis = time.time()
    timerundrow = time.time()
    timerundis = time.time()
    main()