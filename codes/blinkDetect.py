import dlib
import cv2
import numpy as np
import time
from detection import Detection

if __name__ == "__main__":

    # Initialize Detection class
    _detection = Detection()

    # Calibrate
    print("Calibration started")
    # _detection.calibration()
    print("Calibration completed!")
    # End calibration

    # Output video to a file
    _detection.vid_writer = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc(*"DIVX"), 25, (640, 480))
    
    # Start detection
    while(1):
        try:
            # Read frame and resize
            _detection.readAndResizeFrame()
            
            adjusted = _detection.histogram_equalization()
            landmarks = _detection.getLandmarks(adjusted)
            
            # When no face detected
            if landmarks == 0:
                _detection.offAlarm()
                
                # Break if ESC pressed
                if cv2.waitKey(30) & 0xFF == 27:
                    _detection.safeExit() # break
                    
                continue
            # end if
            
            # Check eyes
            eyeStatus = _detection.checkEyeStatus(landmarks)
            _detection.checkBlinkStatus(eyeStatus)
            
            for i in range(0, len(_detection.leftEyeIndex)):
                cv2.circle(_detection.frame, (landmarks[_detection.leftEyeIndex[i]][0], landmarks[_detection.leftEyeIndex[i]][1]), 1, (0, 0, 255), -1, lineType=cv2.LINE_AA)

            for i in range(0, len(_detection.rightEyeIndex)):
                cv2.circle(_detection.frame, (landmarks[_detection.rightEyeIndex[i]][0], landmarks[_detection.rightEyeIndex[i]][1]), 1, (0, 0, 255), -1, lineType=cv2.LINE_AA)
 
            # If drowsy
            if _detection.drowsy:
                print("Drowsy!!!");
                cv2.putText(_detection.frame, "! ! ! DROWSINESS ALERT ! ! !", (70, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                
                # On the alarm
                _detection.onAlarm()
            
            # If not drowsy
            else:
                cv2.putText(_detection.frame, "Blinks : {}".format(_detection.blinkCount), (460, 80), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0,0,255), 2, cv2.LINE_AA)
                _detection.offAlarm()

            # Listen for input
            k = cv2.waitKey(30)
            
            # If 'r' is pressed, reset alarm
            if k == ord('r'):
                _detection.state = 0
                _detection.drowsy = 0
                _detection.offAlarm()

            # If ESC pressed, stop the program
            elif k == 27:
                _detection.safeExit() # break
            
        except Exception as e:
            print('Exception: ' + repr(e))

    _detection.safeExit()