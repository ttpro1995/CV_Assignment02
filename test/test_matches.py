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
from matcher import Matcher

# make sure it can find detector.py file
sys.path.append(os.path.realpath(os.path.dirname(__file__)+"/.."))

from detector import Detector

image_1_path = './image/voi.png'
image_2_path = './image/voi_large.JPG'
class TestMatcher():



    def test_matches_default_sift(self):
        _name = 'test_match_default_sift'
        _file  = './output/'+_name+'.png'
        img1 = cv2.imread(image_1_path)
        img2 = cv2.imread(image_2_path)
        _matcher = Matcher()
        matches, result = _matcher.default_match(img1, img2, 20)
        cv2.imwrite(_file, result)
        upload.imgur(_file, _name)
