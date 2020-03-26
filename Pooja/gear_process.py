import os,cv2,numpy as np,time,math
from envision import crop

class gear_process:

    def __init__(self,image):
        self.image = image

    def __sub_image(self,image, center, theta, width, height):
        """Extract a rectangle from the source image.

        image - source image
        center - (x,y) tuple for the centre point.
        theta - angle of rectangle.
        width, height - rectangle dimensions.
        This function rotates the desired part of the image and crops it and returns the image
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


    def __find_contour(self,col_image=[],type = 0):
        image = col_image.copy()
        num,img,image_parts = 0,[],[]
        lower_limit_area = 7000 if type else 2000
        upper_limit_area = 30000 if type else 15000

        contours, hierarchy = cv2.findContours(image[:,:,1], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
        if contours:
            for c in contours:
                area = cv2.contourArea(c)
                
                if area < lower_limit_area or area > upper_limit_area:
                    continue
                num += 1
                print(" tooth/flank count: " + str(num) + "\t area: " + str(area))

                rect = cv2.minAreaRect(c)
                print(rect)
                width = int(rect[1][0])
                height = int(rect[1][1])
                angle = rect[2]
                print("\t\t\t\t", angle)

                box = cv2.boxPoints(rect)
                box = np.int0(box)
                
                # img = cv2.drawContours(image,[box], -1, (0,255,0), 2)
                M = cv2.moments(box)

                """
                following code is done to find the center of the particular 
                contour which is used ahead in subimage to crop the image
                """

                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                else:
                    cX, cY = 0, 0

                image_patch = self.__sub_image(image, (cX, cY), angle + 90, height, width)
                image_parts.append(image_patch)

            # cv2.imwrite("analysis/"+str(num)+"contour_draw.jpg",img)
        
        return image_parts

    def process_separation(self,contour = False):
        img = self.image
        threshold_value,flank_list,teeth_list = 200,[],[]
        blank_img_flank = np.zeros_like(img)
        blank_img_teeth = np.zeros_like(img)
        img_blur = cv2.medianBlur(img, 29)
        
        crop_img, _ = crop.crop_radial_arc_two_centres(img_blur, centre_x1=400, centre_y1=-320,
                                                     centre_x2=400, centre_y2=-150, radius1=500, radius2=510,
                                                     theta1=230,
                                                     theta2=310)
        green_channel = crop_img[:, :, 1]
        cv2.imwrite("analysis/A2_blur.jpg",green_channel)

        pos1 = np.where(np.logical_and(green_channel <= threshold_value,green_channel>0))
        pos2 = np.where(green_channel >= threshold_value)

        blank_img_flank[pos1] = img[pos1]
        blank_img_teeth[pos2] = img[pos2]

        if contour:
            flank_list = self.__find_contour(col_image = blank_img_flank,type = 1)

            teeth_list = self.__find_contour(col_image = blank_img_teeth,type = 0)
            return flank_list,teeth_list

        flank_list.append(blank_img_flank)
        teeth_list.append(blank_img_teeth)
        return flank_list,teeth_list

    


