# Thai Thien
# 1351040

import pytest
import cv2
import numpy as np
import sys
import os
import upload
# make sure it can find detector.py file
sys.path.append(os.path.realpath(os.path.dirname(__file__)+"/.."))
import util

image_path = './image/cat.jpg'
class TestUtil():

    def test_cv2(self):
        cat = cv2.imread(image_path)
        assert cat.shape == (566,604,3)

    def test_addnoise(self):
        _name = 'test_util_addnoise'
        _file  = './output/'+_name+'.png'
        cat = cv2.imread(image_path)
        noisy_ret = util.add_noise(cat,0.1)
        assert noisy_ret.shape == cat.shape
        cv2.imwrite(_file, noisy_ret)
        upload.imgur(_file, _name)
