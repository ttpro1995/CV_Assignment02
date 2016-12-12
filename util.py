# Thai Thien
# 1351040

import cv2
import random
import numpy as np
def help():
    helpMessage = '''
    + harris image.jpg detect key points using harris algorithm and show the keypoints in original image.
    + blob image.jpg - detect key points using blob algorithm and show the keypoints in original image.
    + dog image.jpg detect key points using DoG Algorithm and show keypoints in original image.
    + m harris sift image1.jpg image2.jpg match and show results of image1 and image2 using Harris detector and SIFT descriptor.
    + m dog sift image1.jpg image2.jpg - match and show results of image1 and image2 using DoG detector and SIFT descriptor.
    + m blob sift image1.jpg image2.jpg - match and show results of image1 and image2 using using Blob detector and SIFT descriptor.
    + m harris lbp image1.jpg image2.jpg - match and show results of image1 and image2 using Harris detector and LBP descriptor.
    + m dog lbp image1.jpg image2.jpg - match and show results of image1 and image2 using DoG detector and LBP descriptor.
    + m blob lbp image1.jpg image2.jpg - match and show results of image1 and image2 using Blob detector and LBP descriptor.
    + h - Display a short description of the program, its command line arguments, and the keys it supports.
    '''
    print (helpMessage)

def incorrect_argv():
    incorrect_argv_message ='''
    Usage:
        <detector> <filename>
        m <detector> <descriptor> <filename1> <filename2>
    '''
    print (incorrect_argv_message)

def add_noise(img, noise):
    '''
    Add some salt and peper
    :param img:
    :param noise:
    :return:
    '''
    r = random.SystemRandom()
    noise += r.uniform(float(noise)/10, float(noise)/1)
    result_image = img.copy()
    noise_mask = np.zeros(result_image.shape,result_image.dtype)
    cv2.randu(noise_mask, np.zeros(3), np.ones(3) * 255 * noise)
    result_image = cv2.add(result_image,noise_mask)
    return result_image

def drawMatches(img1, kp1, img2, kp2, matches, isKnn = False):
    """
    This function takes in two images with their associated
    keypoints, as well as a list of DMatch data structure (matches)
    that contains which keypoints matched in which images.

    An image will be produced where a montage is shown with
    the first image followed by the second image beside it.

    Keypoints are delineated with circles, while lines are connected
    between matching keypoints.

    img1,img2 - Grayscale images
    kp1,kp2 - Detected list of keypoints through any of the OpenCV keypoint
              detection algorithms
    matches - A list of matches of corresponding keypoints through any
              OpenCV keypoint matching algorithm
    """

    # Create a new output image that concatenates the two images together
    # (a.k.a) a montage
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]
    space = (cols1+cols2)/20

    out = np.zeros((max([rows1,rows2]),cols1+cols2+space,3), dtype='uint8')

    # Place the first image to the left
    out[:rows1,:cols1,:] = np.dstack([img1])

    # Place the next image to the right of it
    out[:rows2,cols1+space:cols1+cols2+space,:] = np.dstack([img2])

    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    if (isKnn == False):
        for mat in matches:

            # Get the matching keypoints for each of the images
            img1_idx = mat.queryIdx
            img2_idx = mat.trainIdx

            # x - columns
            # y - rows
            (x1,y1) = kp1[img1_idx].pt
            (x2,y2) = kp2[img2_idx].pt
            x2+= space
            # Draw a small circle at both co-ordinates
            # radius 4
            # colour blue
            # thickness = 1
            cv2.circle(out, (int(x1),int(y1)), 4, (255, 0, 0), 1)
            cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 1)

            # Draw a line in between the two points
            # thickness = 1
            # colour blue
            cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255, 0, 0), 1)
    else:
        for submat in matches:
            for mat in submat:
                # Get the matching keypoints for each of the images
                img1_idx = mat.queryIdx
                img2_idx = mat.trainIdx

                # x - columns
                # y - rows
                (x1, y1) = kp1[img1_idx].pt
                (x2, y2) = kp2[img2_idx].pt
                x2 += space
                # Draw a small circle at both co-ordinates
                # radius 4
                # colour blue
                # thickness = 1
                cv2.circle(out, (int(x1), int(y1)), 4, (255, 0, 0), 1)
                cv2.circle(out, (int(x2) + cols1, int(y2)), 4, (255, 0, 0), 1)

                # Draw a line in between the two points
                # thickness = 1
                # colour blue
                cv2.line(out, (int(x1), int(y1)), (int(x2) + cols1, int(y2)), (255, 0, 0), 1)

    return out

