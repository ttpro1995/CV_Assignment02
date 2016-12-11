import cv2
import sys
import numpy as np
import util
from detector import Detector
from descriptor import Descriptor

HARRIS = 'harris'
BLOB = 'blob'
DOG = 'dog'

def main():

    _detector = Detector()
    descriptor = Descriptor()

    if (len(sys.argv) < 2):
        util.help()
        quit()

    image_name1 = None
    image_name2 = None
    detector_method = None
    descriptor_method = None
    isMatchMode = False
    result_image1 = None

    # Parse argv
    if (sys.argv[1] == 'm'):
        isMatchMode = True
        detector_method = sys.argv[2]
        descriptor_method = sys.argv[3]
        image_name1 = sys.argv[4]
        image_name2 = sys.argv[5]
    elif (sys.argv[1] in [HARRIS, BLOB, DOG]):
        detector_method = sys.argv[1]
        image_name1 = sys.argv[2]
    else:
        util.incorrect_argv()

    print (detector_method)

    image1 = cv2.imread(image_name1)
    if (image_name2 != None):
        image2 = cv2.imread(image_name2)

    if (isMatchMode == False):
        if (detector_method == HARRIS):
            print 'Run harris detector'
            feature, result_image1 = _detector.harris(image1)
        if (detector_method == BLOB):
            print 'Run blob detector'
            keypoints, result_image1 = _detector.blob(image1)
        if (detector_method == DOG):
            print 'Run DOG detector'
            feature, result_image1 = _detector.dog(image1)


    if (result_image1 is not None):
        print('print sth')
        cv2.imwrite('./output/'+image_name1, result_image1)
        cv2.imshow('image',result_image1)
        cv2.waitKey()

if __name__ =='__main__':
    main()