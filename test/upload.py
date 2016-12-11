# Thai Thien
# 1351040

# b18f28cc83c7acf
import pyimgur
import time
def imgur(path, title):
    CLIENT_ID = "b18f28cc83c7acf"
    im = pyimgur.Imgur(CLIENT_ID)
    uploaded_image = im.upload_image(path, title=title+' '+str(time.time()))
    print('\nImgur Title: ',uploaded_image.title)
    print('\nImgur Link: ', uploaded_image.link)
    print('\nImgur Size: ', uploaded_image.size)
    print('\nImgur Type: ', uploaded_image.type)
