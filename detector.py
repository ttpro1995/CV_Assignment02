import cv2
import numpy as np
class Detector:
    def __init__(self):
        pass

    def harris(self, img):
        gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        gray_img = np.float32(gray_img)
        dst = cv2.cornerHarris(gray_img, 2, 3, 0.04)
        result_img = img.copy() # deep copy image
        # result is dilated for marking the corners, not important
        dst = cv2.dilate(dst, None)

        # Threshold for an optimal value, it may vary depending on the image.
        result_img[dst > 0.01 * dst.max()] = [0, 0, 255]
        return (dst, result_img)

    def blob(self):
        pass

    def dog(self):
        pass