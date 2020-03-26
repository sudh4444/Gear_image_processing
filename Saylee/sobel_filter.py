# Saylee Kanitkar
# We use this code to find arc of elipse
import os
import cv2
import numpy as np
from envision.convolution import convolve_sobel

# image = cv2.imread('/home/saylee/Downloads/blue_new.png', 0)

# image = cv2.imread('/home/saylee/Downloads/2020_3_18_18_46_10.jpg', 0)
path = "/home/saylee/Downloads/ok_b2/"
for f in os.listdir(path):
    image = cv2.imread(path + f,0)

    image1 = image[100:250,240:560] #crop elipse area
    cv2.imshow("Gray", image1)
    cv2.waitKey(0)

    # use sobel to find edges of elipse
    sobel = convolve_sobel(img=image1,
                               threshold=170,
                               sobel_kernel_left_right=False,
                               sobel_kernel_right_left=False,
                               sobel_kernel_top_bottom=False,
                               sobel_kernel_bottom_top=True,
                               sobel_kernel_diagonal_top_left=False,
                               sobel_kernel_diagonal_bottom_left=False,
                               sobel_kernel_diagonal_top_right=False,
                               sobel_kernel_diagonal_bottom_right=False)
    sobel = cv2.convertScaleAbs(sobel)
    test_image = np.zeros_like(sobel)
    pos_y, pos_x = np.where(sobel == 255)

    cv2.imshow("sobel",sobel)
    cv2.imwrite("/home/saylee/Downloads/sobel.jpeg",sobel)
    cv2.waitKey(0)

    # kernel3 = np.ones((5,5),np.uint8)
    # closing = cv2.morphologyEx(sobel, cv2.MORPH_CLOSE, kernel3)
    #
    # cv2.imshow('people',closing)
    # cv2.waitKey(0)

    blur = cv2.GaussianBlur(src=sobel, ksize=(7, 7), sigmaX=0)
    cv2.imshow('blur', blur)
    cv2.waitKey(0)

    print(np.count_nonzero(blur))


cv2.destroyAllWindows()