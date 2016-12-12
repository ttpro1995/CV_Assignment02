# Thai Thien
# 1351040

import pytest
import cv2
import sys
import sys, os
import numpy as np
import upload

# make sure it can find matcher.py file
sys.path.append(os.path.realpath(os.path.dirname(__file__)+"/.."))
import util
from matcher import Matcher

# make sure it can find detector.py file
sys.path.append(os.path.realpath(os.path.dirname(__file__)+"/.."))

from detector import Detector

image_1_path = './image/voi.png'
image_2_path = './image/voi_large.JPG'

isUpload = True
class TestMatcher():

    def test_matches__orb(self):
        _name = 'test_match_default_orb'
        _file  = './output/'+_name+'.png'
        img1 = cv2.imread(image_1_path)
        img1 = util.add_noise(img1, 0.2)
        img2 = cv2.imread(image_2_path)
        img2 = util.add_noise(img2, 0.2)
        _matcher = Matcher()
        matches, result = _matcher.orb_match(img1, img2, 20)
        cv2.imwrite(_file, result)
        if (isUpload):
            upload.imgur(_file,_name)

    def test_matches_dog_sift(self):
        _name = 'test_match_default_sift'
        _file  = './output/'+_name+'.png'
        img1 = cv2.imread(image_1_path)
        img1 = util.add_noise(img1, 0.2)
        img2 = cv2.imread(image_2_path)
        img2 = util.add_noise(img2, 0.2)
        _matcher = Matcher()
        matches, result = _matcher.dog_match(img1, img2, 20)
        cv2.imwrite(_file, result)
        if (isUpload):
            upload.imgur(_file,_name)

    def test_matches_harris_sift(self):
        _name = 'test_match_harris_sift'
        _file  = './output/'+_name+'.png'
        img1 = cv2.imread(image_1_path)
        img1 = util.add_noise(img1, 0.2)
        img2 = cv2.imread(image_2_path)
        img2 = util.add_noise(img2, 0.2)
        _matcher = Matcher()
        matches, result = _matcher.harris_match(img1, img2, 50)
        cv2.imwrite(_file, result)
        if (isUpload):
            upload.imgur(_file,_name)


