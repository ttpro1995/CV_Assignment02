# Thai Thien
# 1351040

import pytest
import cv2
import sys
import sys, os
import numpy as np

# make sure it can find detector.py file
sys.path.append(os.path.realpath(os.path.dirname(__file__)+"/.."))

from detector import Detector

image_path = './test/cat.jpg'
class TestDetector():

    def test_cv2(self):
        cat = cv2.imread(image_path)
        assert cat.shape == (566,604,3)

    def test_harris(self):
        cat = cv2.imread(image_path)
        assert cat.shape == (566,604,3)
        _detector = Detector()
        feature, result = _detector.harris(cat)
        assert result.shape == (566, 604, 3)
        assert feature.shape == (566, 604)