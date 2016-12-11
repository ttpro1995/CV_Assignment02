import cv2

class Descriptor:
    def __init__(self):
        pass
    def sift(self, img, keypoint):
        gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        _sift = cv2.SIFT()
        kp, des = kp,des = _sift.compute(gray_img,keypoint)
        return kp, des

    def lbp(self):
        pass
