# Thai Thien
# 1351040

import pytest
import cv2
import numpy as np
image_path = 'cat.jpg'
class TestUtil():

    def test_cv2(self):
        cat = cv2.imread(image_path)
        assert cat.shape == (566,604,3)
