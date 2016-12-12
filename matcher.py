# Thai Thien
# 1351040

import cv2
import numpy as np
import util
from detector import Detector
import random
from lbp import LBP

class Matcher:
    def __init__(self):
        self.SIFT = 'sift'
        self.LBP = 'lbp'
        self._orb = cv2.ORB()
        self._sift = cv2.SIFT()
        self._lbp = LBP()

        self._detector = Detector()

    def orb_match(self, img1, img2, num_drawmatch):
        '''
        Match 2 image use orb detector and sift
        :param img1: first image
        :param img2: second image
        :param num_drawmatch: number of line in result_image
        :return: (matches, result_image)
        '''

        kp1, des1 = self._orb.detectAndCompute(img1, None)
        kp2, des2 = self._orb.detectAndCompute(img2, None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Match descriptors.
        matches = bf.match(des1, des2)

        # Sort them in the order of their distance.
        matches = sorted(matches, key=lambda x: x.distance)
        result_image = util.drawMatches(img1, kp1, img2, kp2, matches[:num_drawmatch])
        return (matches, result_image)

    def dog_match(self, img1, img2, num_drawmatch, type = 'sift'):
        kp1 = self._sift.detect(img1)
        kp2 = self._sift.detect(img2)
        if (type == self.SIFT):
            print ('using SIFT descriptor')
            kp1, des1 = self._sift.compute(img1, kp1)
            kp2, des2 = self._sift.compute(img2, kp2)
        elif(type==self.LBP):
            print ('using LBP descriptor')
            kp1, des1 = self._lbp.compute(img1, kp1)
            kp2, des2 = self._lbp.compute(img2, kp2)

        bf = cv2.BFMatcher()

        matches = bf.knnMatch(des1, des2, k = 2)

        # Sort them in the order of their distance.
        # matches = sorted(matches, key=lambda x: x.distance)
        result_image = util.drawMatches(img1, kp1, img2, kp2, matches[:num_drawmatch], isKnn=True)
        return (matches, result_image)

    def harris_match(self, img1, img2, num_drawmatch, type = 'sift'):
        # find the keypoints and descriptors with SIFT
        kp1, _ = self._detector.harris(img1)
        kp2, _ = self._detector.harris(img2)

        if (type == self.SIFT):
            print ('using SIFT descriptor')
            kp1, des1 = self._sift.compute(img1, kp1)
            kp2, des2 = self._sift.compute(img2, kp2)
        elif(type==self.LBP):
            print ('using LBP descriptor')
            kp1, des1 = self._lbp.compute(img1, kp1)
            kp2, des2 = self._lbp.compute(img2, kp2)

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)

        # Sort them in the order of their distance.
        # matches = sorted(matches, key=lambda x: x.distance)
        random.shuffle(matches)
        if (num_drawmatch < 0):
            mini_match = matches
        else:
            mini_match =  matches[:num_drawmatch]
        result_image = util.drawMatches(img1, kp1, img2, kp2,mini_match, isKnn=True)
        return (matches, result_image)