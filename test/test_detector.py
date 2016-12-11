# Thai Thien
# 1351040

import pytest
import cv2
import sys
import sys, os
import numpy as np
import upload

# make sure it can find detector.py file
sys.path.append(os.path.realpath(os.path.dirname(__file__)+"/.."))

from detector import Detector

image_path = './test/cat.jpg'
blob_path = './test/blobsample2.png'
class TestDetector():

    def test_cv2(self):
        cat = cv2.imread(image_path)
        assert cat.shape == (566,604,3)

    def test_harris(self):
        _name = 'test_harris'
        _file  = './output/'+_name+'.png'
        cat = cv2.imread(image_path)
        assert cat.shape == (566,604,3)
        _detector = Detector()
        feature, result = _detector.harris(cat)
        cv2.imwrite(_file, result)
        upload.imgur(_file, _name)
        assert result.shape == (566, 604, 3)
        assert feature.shape == (566, 604)

    def test_blob(self):
        _name = 'test_blob'
        _file  = './output/'+_name+'.png'
        img = cv2.imread(blob_path)
        _detector = Detector()
        feature, result = _detector.blob(img)
        cv2.imwrite(_file, result)
        upload.imgur(_file, _name)
        assert result.shape == img.shape # result should have same dimension with input

    def test_dog(self):
        _name = 'test_dog'
        _file  = './output/'+_name+'.png'
        img = cv2.imread(image_path)
        _detector = Detector()
        feature, result = _detector.dog(img)
        cv2.imwrite(_file, result)
        upload.imgur(_file, _name)
        assert result.shape == img.shape # result should have same dimension with input