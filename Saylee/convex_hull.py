# Saylee Kanitkar
# dont use this code

import cv2
import numpy as np
import math
import os

image = cv2.imread('/media/saylee/Work/oil2/image116.jpg', 0)

# image = cv2.imread('/home/saylee/Downloads/blue_new.png', 0)
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

kernel1 = np.ones((2,2),np.uint8)
closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel1)
cv2.imshow('closing',closing)
cv2.waitKey(0)

ret,thresh = cv2.threshold(closing,105,255,0)
cv2.imshow('thresh again',thresh)
cv2.waitKey(0)

kernel3 = np.ones((1,1),np.uint8)
closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel3)

# cv2.imshow('people',closing)
# cv2.waitKey(0)

if 0:
    _, contours,hierarchy = cv2.findContours(closing, cv2.RETR_EXTERNAL, 2)
    count1=0
    cnt = contours[0]
    for cnt in contours:
        M = cv2.moments(cnt)
        area = cv2.contourArea(cnt)
        if area < 1000:
            continue
        print(area)
        if (area%9000)>900:
            # print("Count is" + str(math.ceil(area / 9000.0)))
            count1 += math.ceil(area / 9000.0)

        else:
            # print("Count is" + str(math.floor(area / 9000.0)))
            count1 += math.floor(area / 9000.0)
        perimeter = cv2.arcLength(cnt,True)

        epsilon = 0.1*cv2.arcLength(cnt,True)
        approx = cv2.approxPolyDP(cnt,epsilon,True)

        k = cv2.isContourConvex(cnt)

        x,y,w,h = cv2.boundingRect(cnt)
        subImg = image[y: y + h, x: x + w]
        # cv2.imshow("result", subImg)
        cv2.imwrite(filename="sub-{}.jpg".format(i), img=subImg)
        i += 1
        image = cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
        # cv2.waitKey(0)

        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        image = cv2.drawContours(image,[box],0,(255,255,255),2)
    txt = "Total count is" + ' ' + str(count1)
    cv2.putText(image, txt,(50,50),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,0,255),2)
    print(count1)
    cv2.imshow("result", image)
    cv2.waitKey(0)
    capagain = 255 * np.ones(image.shape, dtype=np.uint8)
    cv2.putText(capagain, "Capturing...", (350,350),cv2.FONT_HERSHEY_SIMPLEX,2.0,(0,0,255),2)
    cv2.imshow("result",capagain)
    cv2.waitKey(0)

if 1:
    im2, contours, hierarchy = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # create hull array for convex hull points
    hull = []
    count1 = 0
    cnt = contours[0]
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 100:
            continue
        print(area)

    # calculate points for each contour
    for i in range(len(contours)):
        # creating convex hull object for each contour
        hull.append(cv2.convexHull(contours[i], False))

        # create an empty black image
        drawing = np.zeros((thresh.shape[0], thresh.shape[1], 3), np.uint8)
        cv2.imshow('drawing', drawing)
        cv2.waitKey(0)

    # draw contours and hull points
    for i in range(len(contours)):
        color_contours = (0, 255, 0)  # green - color for contours
        color = (255, 0, 0)  # blue - color for convex hull
        # area = cv2.contourArea(cnt)
        # print(area)
        # draw ith contour
        cv2.drawContours(drawing, contours, i, color_contours, 1, 8, hierarchy)
        # draw ith convex hull object
        cv2.drawContours(drawing, hull, i, color, 1, 8)

    cv2.imshow('convex hull', drawing)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




