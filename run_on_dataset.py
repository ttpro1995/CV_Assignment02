import cv2
import numpy as np
from matcher import Matcher
import test.upload as upload
HARRIS = 'harris'
BLOB = 'blob'
DOG = 'dog'
ORB = 'orb'
SIFT = 'sift'
LBP = 'lbp'

output_prefix = 'output/'

def run_match(detector_method, descriptor_method, image1, image2, filename, isUpload = True):
    _matcher = Matcher()
    if (descriptor_method == SIFT):
        if (detector_method == HARRIS):
            matches, result_image1 = _matcher.harris_match(image1, image2, 30)
        if (detector_method == ORB):
            matches, result_image1 = _matcher.orb_match(image1, image2, 30)
        if (detector_method == DOG):
            matches, result_image1 = _matcher.dog_match(image1, image2, 30)
        if (detector_method == BLOB):
            matches, result_image1 = _matcher.blob_match(image1, image2, 30)
    if (descriptor_method == LBP):
        if (detector_method == HARRIS):
            matches, result_image1 = _matcher.harris_match(image1, image2, 30, type=LBP)
        if (detector_method == ORB):
            matches, result_image1 = _matcher.orb_match(image1, image2, 30, type=LBP)
        if (detector_method == DOG):
            matches, result_image1 = _matcher.dog_match(image1, image2, 30, type=LBP)
        if (detector_method == BLOB):
            matches, result_image1 = _matcher.blob_match(image1, image2, 30, type=LBP)

    title = detector_method+'_'+descriptor_method+'_'+filename
    filename = output_prefix + title
    cv2.imwrite(filename, result_image1)
    print ('saved %s' %(filename))
    if (isUpload):
        upload.imgur(filename, title)
    return matches, result_image1

def run():
    print('RUNNING on meow data set')
    prefix = './image/meowdataresized/'

    voi1_path = prefix+ 'voi.png'
    voi2_path = prefix+ 'voi_large.JPG'

    voi1 = cv2.imread(voi1_path)
    voi2 = cv2.imread(voi2_path)

    # run_match(HARRIS, SIFT, voi1, voi2, 'voi1_2.png')

    chuot1_path = prefix + 'chuot1.jpg'
    chuot2_path = prefix + 'chuot1.jpg'
    chuot3_path = prefix + 'chuot1.jpg'

    dau1_path = prefix + 'dau1.jpg'
    dau2_path = prefix + 'dau2.jpg'
    dau3_path = prefix + 'dau3.jpg'
    dau4_path = prefix + 'dau4.jpg'

    keyboard1_path = prefix + 'keyboard1.jpg'
    keyboard2_path = prefix + 'keyboard2.jpg'
    keyboard3_path = prefix + 'keyboard3.jpg'
    keyboard4_path = prefix + 'keyboard4.jpg'

    nuoc1_path = prefix + 'nuoc1.jpg'
    nuoc2_path = prefix + 'nuoc2.jpg'

    pen1_path = prefix + 'pen1.jpg'
    pen2_path = prefix + 'pen2.jpg'


    nuoc1 = cv2.imread(nuoc1_path)
    nuoc2 = cv2.imread(nuoc2_path)

    pen1 = cv2.imread(pen1_path)
    pen2 = cv2.imread(pen2_path)

    chuot1 = cv2.imread(chuot1_path)
    chuot2 = cv2.imread(chuot2_path)


    dau1 = cv2.imread(dau1_path)
    dau2 = cv2.imread(dau2_path)

    keyboard1 = cv2.imread(keyboard1_path)
    keyboard2 = cv2.imread(keyboard2_path)


    run_match(HARRIS, SIFT, keyboard1, keyboard2, 'keyboard1_2.png')
    run_match(HARRIS, SIFT, dau1, dau2, 'dau1_2.png')
    run_match(HARRIS, SIFT, chuot1, chuot2, 'chuot1_2.png')

    run_match(HARRIS, SIFT, nuoc1, nuoc2, 'nuoc1_2.png')
    run_match(HARRIS, SIFT, pen1, pen2, 'pen1_2.png')
    run_match(BLOB, SIFT, keyboard1, keyboard2, 'keyboard1_2.png')
    run_match(BLOB, SIFT, dau1, dau2, 'dau1_2.png')
    run_match(BLOB, SIFT, chuot1, chuot2, 'chuot1_2.png')
    run_match(BLOB, SIFT, pen1, pen2, 'pen1_2.png')
    run_match(BLOB, SIFT, nuoc1, nuoc2, 'nuoc1_2.png')

    run_match(DOG, SIFT, keyboard1, keyboard2, 'keyboard1_2.png')
    run_match(DOG, SIFT, dau1, dau2, 'dau1_2.png')
    run_match(DOG, SIFT, chuot1, chuot2, 'chuot1_2.png')
    run_match(DOG, SIFT, pen1, pen2, 'pen1_2.png')
    run_match(DOG, SIFT, nuoc1, nuoc2, 'nuoc1_2.png')

    run_match(HARRIS, LBP, keyboard1, keyboard2, 'keyboard1_2.png')
    run_match(HARRIS, LBP, dau1, dau2, 'dau1_2.png')
    run_match(HARRIS, LBP, chuot1, chuot2, 'chuot1_2.png')
    run_match(HARRIS, LBP, pen1, pen2, 'pen1_2.png')
    run_match(HARRIS, LBP, nuoc1, nuoc2, 'nuoc1_2.png')

    run_match(BLOB, LBP, keyboard1, keyboard2, 'keyboard1_2.png')
    run_match(BLOB, LBP, dau1, dau2, 'dau1_2.png')
    run_match(BLOB, LBP, chuot1, chuot2, 'chuot1_2.png')
    run_match(BLOB, LBP, pen1, pen2, 'pen1_2.png')
    run_match(BLOB, LBP, nuoc1, nuoc2, 'nuoc1_2.png')

    run_match(DOG, LBP, keyboard1, keyboard2, 'keyboard1_2.png')
    run_match(DOG, LBP, dau1, dau2, 'dau1_2.png')
    run_match(DOG, LBP, chuot1, chuot2, 'chuot1_2.png')
    run_match(DOG, LBP, pen1, pen2, 'pen1_2.png')
    run_match(DOG, LBP, nuoc1, nuoc2, 'nuoc1_2.png')

    print('DONE')


run()