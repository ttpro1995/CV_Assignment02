import cv2
import numpy as np
import util
class Matcher:
    def __init__(self):
        self._orb = cv2.ORB()
        self._sift = cv2.SIFT()
        self._bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def default_match(self, img1, img2, num_drawmatch):
        '''
        Match 2 image use orb detector and sift
        :param img1: first image
        :param img2: second image
        :param num_drawmatch: number of line in result_image
        :return: (matches, result_image)
        '''
        # find the keypoints and descriptors with SIFT
        kp1, des1 = self._orb.detectAndCompute(img1, None)
        kp2, des2 = self._orb.detectAndCompute(img2, None)

        # Match descriptors.
        matches = self._bf.match(des1, des2)

        # Sort them in the order of their distance.
        matches = sorted(matches, key=lambda x: x.distance)
        result_image = util.drawMatches(img1, kp1, img2, kp2, matches[:num_drawmatch])
        return (matches, result_image)
