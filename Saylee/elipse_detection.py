# Saylee Kanitkar
# this code is not working
import cv2
import numpy as np

image = cv2.imread('/home/saylee/Downloads/blue_new.png', 0)

# image = cv2.imread('/home/saylee/Downloads/2020_3_18_18_46_10.jpg', 0)
print(image.shape)
cv2.imshow("Gray",image)
cv2.waitKey(0)

ret, thresh = cv2.threshold(image, 240, 255, 0)
cv2.imshow("Binary",thresh)
cv2.waitKey(0)

kernel = np.ones((1, 1), np.uint8)
dilation = cv2.dilate(thresh, kernel, iterations=5)
cv2.imshow('dilation',dilation)
cv2.waitKey(0)

edged = cv2.Canny(dilation, 10, 200)
cv2.imshow('edged',edged)
cv2.waitKey(0)

blur = cv2.GaussianBlur(src=edged,ksize=(3, 3),sigmaX=0)
cv2.imshow('blur',blur)
cv2.waitKey(0)

cv2.destroyAllWindows()
