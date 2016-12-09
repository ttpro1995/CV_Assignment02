
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