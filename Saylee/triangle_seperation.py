# Saylee Kanitkar
# this code is used to process area below teeth
# this code is for regular images
# dont use this code
import cv2
import numpy as np
import math
import os
from envision import crop


def triangle_seperation(gray_image1):
    """
    triangle seperation
    :param image:
    :return:
    """

    # showimage("gray_imge", gray_image)
    cv2.imshow("Gray",gray_image1)
    # ret, gray_image = cv2.threshold(gray_image1, 50, 255,cv2.THRESH_BINARY_INV)
    # ret, gray_image = cv2.threshold(gray_image1, 60, 255, 0)

    # gray_image = gray_image1.copy()
    gray_image = np.zeros(gray_image1.shape,dtype=np.uint8)
    showimage("black image", gray_image)


    # pos = np.where(gray_image1 < 80)
    # gray_image[pos] = [255]
    pos = np.where(gray_image1 < 80)
    gray_image[pos] = [255]

    print(gray_image.shape)
    showimage("gray_imge", gray_image)

    gray_image, _ = crop.crop_radial_arc_two_centres(gray_image, centre_x1=390, centre_y1=-80,
                                                     centre_x2=380, centre_y2=-90, radius1=550, radius2=675,
                                                     theta1=230,
                                                     theta2=310)

    showimage("gray_image_triangle", gray_image)
    contours, hierarchy = cv2.findContours(gray_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2:]
    num = 0
    if contours:
        for c in contours:
            area = cv2.contourArea(c)
            if area < 100 or area > 100000:
                continue
            num += 1
            print(" tooth count: " + str(num) + "\t area: " + str(area))
            x, y, w, h = cv2.boundingRect(c)

            img = cv2.rectangle(gray_image1, (x, y), (x + w, y + h), (255, 0, 0), 2)
            showimage("rectangle", img)

            rect = cv2.minAreaRect(c)
            rows, cols = gray_image.shape
            width = int(rect[1][0])
            height = int(rect[1][1])
            center = rect[0]
            angle = rect[2]
            print("\t\t\t\t", angle)

            box = cv2.boxPoints(rect)
            box = np.int0(box)

            text = str(num)
            col1 = (0, 0, 0)
            rot = cv2.getRotationMatrix2D(center, angle - 90, 1)

            M = cv2.moments(box)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = 0, 0

            image_patch = sub_image(gray_image1, (cX, cY), angle + 90, height, width)
            showimage('A1_original_img_' + str(num), image_patch)
            # blue_img = image_patch[:,:,0]
            # showimage("processed_img" + str(num), blue_img)
            # dent_detection(blue_img, num, angle)

    return 'Triangles separated successfully'


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


def main():
        image = cv2.imread("/home/saylee/Downloads/green.png",0)
        triangle_seperation(image)


if __name__ == "__main__":
    main()