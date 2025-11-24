# camera.py
import cv2
import threading
import atexit

class Camera:
    def __init__(self, src=0, width=640, height=480):
        self.src = src
        self.cap = cv2.VideoCapture(self.src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.lock = threading.Lock()
        atexit.register(self.release)

    def read(self):
        with self.lock:
            if not self.cap or not self.cap.isOpened():
                self.cap = cv2.VideoCapture(self.src)
            ret, frame = self.cap.read()
            return ret, frame

    def release(self):
        with self.lock:
            if self.cap and self.cap.isOpened():
                self.cap.release()
