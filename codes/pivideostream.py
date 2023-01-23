import cv2
from threading import Thread
from picamera.array import PiRGBArray
from picamera import PiCamera

class PiVideoStream:
    def __init__(self, resolution=(640,480), framerate=30):
        # initialize the camera
        self.camera = PiCamera()
        
        # set camera parameters
        self.camera.resolution = resolution
        self.camera.framerate = framerate
        
        # initialize the stream
        self.rawCapture = PiRGBArray(self.camera, size=resolution)
        self.stream = self.camera.capture_continuous(self.rawCapture,
                                                format="bgr",
                                                use_video_port=False)
        
        # initialize the frame and the variable used to indicate
        # if the thread should be stopped
        self.frame = None
        self.stopped = False
        
    def start(self):
        # start the thread to reaed frames from the video stream
        Thread(target=self.update, args=()).start()
        return self
    
    def update(self):
        # loop infinitely until the thread is stopped
        for f in self.stream:
            self.frame = f.array
            self.rawCapture.truncate(0)
            
            # Show the frame
            cv2.imshow("Blink Detection Demo", self.frame)
            #cv2.waitKey(30)
            if cv2.waitKey(30) & 0xFF == 27:
                self.stop()
                
            if self.stopped:
                print("stopped")
                self.stream.close()
                self.camera.close()
                return

    def read(self):
        return self.frame
    
    def stop(self):
        self.stopped = True