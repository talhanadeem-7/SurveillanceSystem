import cv2

class VideoLoader:
    def __init__(self, source):
        self.cap = cv2.VideoCapture(source)
        
    def get_frame(self):
        ret, frame = self.cap.read()
        return ret, frame

    def get_metadata(self):
        return {
            "fps": self.cap.get(cv2.CAP_PROP_FPS),
            "width": int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        }
    
    def release(self):
        self.cap.release()