import cv2
import numpy as np

#read image
# src = cv2.imread('/media/saylee/Work/Allygrow/Gear_image_processing/IMAGES/OK_PART_RPI-B/ok_cam1/2020_3_18_18_51_56.jpg', cv2.IMREAD_UNCHANGED)
# src = cv2.imread('/home/saylee/Downloads/2020_3_18_18_46_10.jpg', cv2.IMREAD_UNCHANGED)
src = cv2.imread("/home/saylee/Downloads/B2_inverted/2020_3_18_14_45_38.jpg")

w, h, c = src.shape
print(w)
print(h)
print(c)

# extract red channel
red_channel = src[:,:,2]
green_channel = src[:,:,1]
blue_channel = src[:,:,0]

# create empty image with same shape as that of src image
red_img = np.zeros((w,h,1))
green_img = np.zeros((w,h,1))
blue_img = np.zeros((w,h,1))

#assign the red channel of src to empty image
# red_img[:,:,0] = red_channel
# red_img[:,:,1] = red_channel
red_img[:,:,0] = red_channel

green_img[:,:,0] = green_channel
# green_img[:,:,1] = green_channel
# green_img[:,:,2] = green_channel

blue_img[:,:,0] =blue_channel
# blue_img[:,:,1] =blue_channel
# blue_img[:,:,2] =blue_channel

#save image
cv2.imshow('RED',red_img)
cv2.imwrite('/home/saylee/Downloads/red.png',red_img)

cv2.imshow('BLUE',blue_img)
cv2.imwrite('/home/saylee/Downloads/blue.png',blue_img)

cv2.imshow('GREEN',green_img)
cv2.imwrite('/home/saylee/Downloads/green.png',green_img)

cv2.waitKey(0)
cv2.destroyAllWindows()