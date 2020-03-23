import numpy as np
import time
import os
import math
import cv2
from envision import crop
from envision.convolution import convolve_sobel

from teeth_seperation import teeth_seperation_method1

ind = 0

def showimage(name,image):
    # return
    # cv2.imwrite("output/" + str(ind) + str(name) +".jpg", image)
    cv2.imshow(str(name),image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def sub_image(image, center, theta, width, height):
    """Extract a rectangle from the source image.

    image - source image
    center - (x,y) tuple for the centre point.
    theta - angle of rectangle.
    width, height - rectangle dimensions.
    """
    if 45 < theta <= 90:
        theta = theta - 90
        width, height = height, width

    theta *= math.pi / 180  # convert to rad
    v_x = (math.cos(theta), math.sin(theta))
    v_y = (-math.sin(theta), math.cos(theta))
    s_x = center[0] - v_x[0] * (width / 2) - v_y[0] * (height / 2)
    s_y = center[1] - v_x[1] * (width / 2) - v_y[1] * (height / 2)
    mapping = np.array([[v_x[0], v_y[0], s_x], [v_x[1], v_y[1], s_y]])

    return cv2.warpAffine(image, mapping, (width, height), flags=cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)


def between_flank_part(color_image):
    """

    :param color_image:
    :return:
    """

    showimage("color_img", color_image)
    gray_img = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray_img, 9)
    original_image = cv2.equalizeHist(blur)
    gray_image = original_image.copy()
    showimage('gray', gray_image)
    pos = np.where(gray_image < 180)
    gray_image[pos] = [0]

    showimage('gray_image', gray_image)

    crop1, _ = crop.crop_radial_arc_two_centres(gray_image, centre_x1=400, centre_y1=-300,
                                                centre_x2=400, centre_y2=-170, radius1=410, radius2=610,
                                                theta1=230,
                                                theta2=310)
    showimage('crop1', crop1)

    contours, hierarchy = cv2.findContours(crop1, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2:]
    n = 0
    if contours:
        for c in contours:
            area = cv2.contourArea(c)
            if area < 3000 or area > 15000:
                continue

            x, y, w, h = cv2.boundingRect(c)

            # saylee kanitkar
            gray_three = cv2.merge([crop1, crop1, crop1])
            img = cv2.rectangle(gray_three, (x, y), (x + w, y + h), (255, 0, 0), 2)
            showimage("rectangle", img)
            n += 1
            rect = cv2.minAreaRect(c)

            rows, cols = gray_image.shape
            width = int(rect[1][0])
            print(" tooth count: " + str(n) + "\t area: " + str(area))

            height = int(rect[1][1])
            center = rect[0]
            angle = rect[2]
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            rot = cv2.getRotationMatrix2D(center, angle - 90, 1)
            img = cv2.warpAffine(color_image, rot, (rows, cols))

            M = cv2.moments(box)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = 0, 0

            image_patch = sub_image(color_image, (cX, cY), angle + 90, height, width)
            showimage('original_img' + str(n), image_patch)



if __name__ == "__main__":
    path = "/media/pooja/G-drive/My-repository/Allygrow/18-3-20/RPI-B/cam1_original/"
    # path = "/media/pooja/G-drive/My-repository/Allygrow/6FM/"
    file_list = []
    file_list = os.listdir(str(path))

    for img in file_list:
        if img.endswith(".jpg"): #2020_3_5_18_54_25
            ind +=1
            image_name = img
            white_pixel_list = []
            print ("imagename", image_name)
            start_time = time.time()
            original_image = cv2.imread(path+img)
            showimage("A0_original", original_image)
            between_flank_part(original_image)
            end_time = time.time()
            print ("time taken:", end_time - start_time)
            # break