import numpy as np
import dlib
import time
import cv2
from threading import Thread
from pivideostream import PiVideoStream
from buzzer import buzzerAlert
from scipy.spatial import distance as dist

class Detection:
    def __init__(self):
        self.blinkCount = 0
        self.blinkTime = 0.15 #150ms
        self.drowsy = 0
        self.drowsyLimit = 0
        self.drowsyTime = 1.0  #1200ms
        self.FACE_DOWNSAMPLE_RATIO = 1.5
        self.falseBlinkLimit = 0
        self.GAMMA = 1.5
        self.invGamma = 1.0/self.GAMMA
        self.RESIZE_HEIGHT = 460
        self.state = 0
        self.table = np.array([((i / 255.0) ** self.invGamma) * 255 for i in range(0, 256)]).astype("uint8")
        self.thresh = 0.3
        self.validFrames = 0
        
        self.cascade = "../models/haarcascade_frontalface_default.xml"
        self.modelPath = "../models/shape_predictor_70_face_landmarks.dat"
        
        self.stopThread = False
        self.alarmThread = Thread()
        
        self.frame = None
        self.vid_writer = cv2.VideoWriter()
        
        self.leftEyeIndex = [36, 37, 38, 39, 40, 41]
        self.rightEyeIndex = [42, 43, 44, 45, 46, 47]
        
        # Initialize PiCamera
        print('Starting PiCamera ......')
        self.vs = PiVideoStream().start()
        time.sleep(2)
        
        self.detector = cv2.CascadeClassifier(self.cascade)
        self.predictor = dlib.shape_predictor(self.modelPath)

    def readFrame(self):
        self.frame = self.vs.read()
        
    def readAndResizeFrame(self):
        self.readFrame()
        
        height, width = self.frame.shape[:2]
        
        self.IMAGE_RESIZE = np.float32(height)/self.RESIZE_HEIGHT
        self.frame = cv2.resize(self.frame, None, 
                            fx = 1/self.IMAGE_RESIZE, 
                            fy = 1/self.IMAGE_RESIZE, 
                            interpolation = cv2.INTER_LINEAR)
        
    def gamma_correction(self):
        return cv2.LUT(self.frame, self.table)

    def histogram_equalization(self):
        gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        return cv2.equalizeHist(gray)

    def eye_aspect_ratio(self, eye):
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)

        return ear

    def checkEyeStatus(self, landmarks):
        mask = np.zeros(self.frame.shape[:2], dtype = np.float32)
        
        hullLeftEye = []
        for i in range(0, len(self.leftEyeIndex)):
            hullLeftEye.append((landmarks[self.leftEyeIndex[i]][0], landmarks[self.leftEyeIndex[i]][1]))

        cv2.fillConvexPoly(mask, np.int32(hullLeftEye), 255)

        hullRightEye = []
        for i in range(0, len(self.rightEyeIndex)):
            hullRightEye.append((landmarks[self.rightEyeIndex[i]][0], landmarks[self.rightEyeIndex[i]][1]))


        cv2.fillConvexPoly(mask, np.int32(hullRightEye), 255)

        leftEAR = self.eye_aspect_ratio(hullLeftEye)
        rightEAR = self.eye_aspect_ratio(hullRightEye)

        ear = (leftEAR + rightEAR) / 2.0

        eyeStatus = 1          # 1 -> Open, 0 -> closed
        if (ear < self.thresh):
            eyeStatus = 0

        return eyeStatus  

    def checkBlinkStatus(self, eyeStatus):
        if(self.state >= 0 and self.state <= self.falseBlinkLimit):
            if(eyeStatus):
                self.state = 0

            else:
                self.state += 1

        elif(self.state >= self.falseBlinkLimit and self.state < self.drowsyLimit):
            if(eyeStatus):
                self.blinkCount += 1 
                self.state = 0

            else:
                state += 1

        else:
            if(eyeStatus):
                self.state = 0
                self.drowsy = 1
                self.blinkCount += 1

            else:
                self.drowsy = 1

    def getLandmarks(self, im):
        imSmall = cv2.resize(im, None, 
                                fx = 1.0/self.FACE_DOWNSAMPLE_RATIO, 
                                fy = 1.0/self.FACE_DOWNSAMPLE_RATIO, 
                                interpolation = cv2.INTER_LINEAR)

        rects = self.detector.detectMultiScale(imSmall, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
        if len(rects) == 0:
            return 0

        arects = rects[0];
        newRect = dlib.rectangle(int(rects[0][0] * self.FACE_DOWNSAMPLE_RATIO),
                                 int(rects[0][1] * self.FACE_DOWNSAMPLE_RATIO),
                                 int((rects[0][2] + rects[0][0]) * self.FACE_DOWNSAMPLE_RATIO),
                                 int((rects[0][3] + rects[0][1]) * self.FACE_DOWNSAMPLE_RATIO)
                                )

        points = []
        [points.append((p.x, p.y)) for p in self.predictor(im, newRect).parts()]
        
        return points

    def doNoFaceDetected(self):    
        cv2.putText(self.frame, "Unable to detect face, Please check proper lighting", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(self.frame, "or decrease FACE_DOWNSAMPLE_RATIO", (10, 50), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)    
        self.validFrames = self.validFrames - 1

    def onAlarm(self):
        # Disable the stop flag
        self.stopThread = False
        
        # Check if alarm queue is running
        if not self.alarmThread.is_alive():
            # Start the queue
            self.alarmThread = Thread(target=buzzerAlert, args=(lambda: self.stopThread,))
            self.alarmThread.daemon = True
            self.alarmThread.start()

    def offAlarm(self):
        self.stopThread = True
        
    def calibration(self):
        totalTime = 0
        dummyFrames = 100
        
        while(self.validFrames < dummyFrames):
            self.validFrames += 1
            t = time.time()
            
            # Read frame and resize
            self.frame = self.readAndResizeFrame()

            adjusted = self.histogram_equalization()

            landmarks = self.getLandmarks(adjusted)
            timeLandmarks = time.time() - t

            # If face detection failed
            if landmarks == 0:
                self.doNoFaceDetected()
                
                # Break if ESC pressed
                if cv2.waitKey(30) & 0xFF == 27:
                    self.safeExit()#break

            else:
                totalTime += timeLandmarks
                print('debug: calibration in progress', totalTime,
                      "Valid frame:", self.validFrames, "out of", dummyFrames)
        
        ## Print
        spf = totalTime/dummyFrames
        print("Current SPF (seconds per frame) is {:.2f} ms".format(spf * 1000))

        if spf != 0:
            self.drowsyTime/spf
            
        if spf != 0:
            self.blinkTime/spf

        print("drowsy limit: {}, false blink limit: {}".format(self.drowsyLimit, self.falseBlinkLimit))
        print("frame shape 1: " + str(self.frame.shape[1]))
        print("frame shape 0: " + str(self.frame.shape[0]))

    def safeExit(self):
        # Release and stop all
        print("safe exiting")
        self.offAlarm()
        self.vs.stop()
        self.vid_writer.release()
        cv2.destroyAllWindows()
        
        time.sleep(1)
        quit()
