import numpy as np
import time
import os
import math
import cv2
from envision import crop
from envision.convolution import convolve_sobel

ind = 0

def show_image(name,image):
    cv2.imwrite("output/" + str(ind) + str(name) +".jpg", image)
    """
    cv2.imshow(str(name),image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """

def flank_seperation(image):
    """
    Flank area
    :param:
    :return:
    """
    color_img = image.copy()

    green_channel = color_img[:, :, 1]
    show_image("A2_green_channel", green_channel)

    image_new = green_channel.copy()
    green_channel = cv2.medianBlur(green_channel, 39)
    show_image("A2_blur", green_channel)


    image_new[np.where(green_channel >= 180)] = [0]
    show_image("A3_patch1", image_new)

    color_img[np.where(green_channel >= 180)] = [0, 0, 0]
    show_image("A4_patch2", color_img)

    crop_img, _ = crop.crop_radial_arc_two_centres(color_img, centre_x1=400, centre_y1=-320,
                                              centre_x2=400, centre_y2=-200, radius1=500, radius2=650,
                                              theta1=235,
                                              theta2=310)
    show_image("A5_crop", crop_img)

    return




if __name__ == "__main__":
    path = "/media/pooja/G-drive/My-repository/Allygrow/18-3-20/RPI-B/cam1_original/"
    # path = "/media/pooja/G-drive/My-repository/Allygrow/18-3-20/RPI-C/invert/"
    # path = "/media/pooja/G-drive/My-repository/Allygrow/6FM/"
    file_list = []
    file_list = os.listdir(str(path))

    for img in file_list:
        if img.endswith(".jpg"):
            ind +=1
            image_name = img
            white_pixel_list = []
            print ("imagename", image_name)
            start_time = time.time()
            original_image = cv2.imread(path+img)
            show_image("A0_original", original_image)

            flank_seperation(original_image)

            end_time = time.time()
            print ("time taken:", end_time - start_time)
            break

