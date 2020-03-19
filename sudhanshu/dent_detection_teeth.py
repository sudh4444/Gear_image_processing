import numpy as np,time,os,math,cv2
from envision import crop
from envision.convolution import convolve_sobel

ind = 0
index_count = 0

def save_image(name,image):
    cv2.imwrite("analysis/"+str(ind)+"_"str(index_count)+"_"+str(name)+".jpg",image)
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


def teeth_seperation_method1(color_image):
    """
    teeth seperation
    :param image:
    :return:
    """
    gray_img = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray_img, 3)
    original_image = cv2.equalizeHist(blur)
    gray_image = original_image.copy()

    pos = np.where(gray_image < 180)
    gray_image[pos] = [0]
    pos = np.where(gray_image > 180)
    gray_image[pos] = [255]
    save_image("teeth manipulation",gray_image)

    gray_image, _ = crop.crop_radial_arc_two_centres(gray_image, centre_x1=400, centre_y1=-270,
                                                 centre_x2=400, centre_y2=-280, radius1=450, radius2=650,
                                                 theta1=230,
                                                 theta2=310)
    
    contours, hierarchy = cv2.findContours(gray_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2:]
    num = 0
    if contours:
        for c in contours:
            area = cv2.contourArea(c)
            if area < 10000 or area > 100000:
                continue
            num += 1
            print(" tooth count: " + str(num) + "\t area: " + str(area))
            rect = cv2.minAreaRect(c)
            width = int(rect[1][0])
            height = int(rect[1][1])
            angle = rect[2]
            print("\t\t\t\t", angle)

            box = cv2.boxPoints(rect)
            box = np.int0(box)

            M = cv2.moments(box)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = 0, 0

            image_patch = sub_image(color_image, (cX, cY), angle + 90, height, width)
            save_image('A1_original_img_' + str(num), image_patch)
            blue_img = image_patch[:,:,0]
            # save_image("processed_img" + str(num), blue_img)
            dent_detection(blue_img, num, angle)

        return 'Teeth seperated successfully'

def teeth_seperation_method2(original_img):
    new_img = original_img.copy()

    gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray_img, 3)
    original_image = cv2.equalizeHist(blur)
    gray_image = original_image.copy()
    pos = np.where(gray_image < 180)
    gray_image[pos] = [0]
    pos = np.where(gray_image > 180)
    gray_image[pos] = [255]
    gray_image, _ = crop.crop_radial_arc_two_centres(gray_image, centre_x1=400, centre_y1=-270,
                                                     centre_x2=400, centre_y2=-280, radius1=450, radius2=650,
                                                     theta1=230,
                                                     theta2=310)
    # save_image("gray_imge", gray_image)
    contours, hierarchy = cv2.findContours(gray_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2:]
    for cnt in contours:
        contourArea = cv2.contourArea(cnt)
        area = cv2.contourArea(cnt)
        if area < 10000 or area > 100000:
            continue
        x, y, w, h = cv2.boundingRect(cnt)

        ROI = new_img[y:y + h, x:x + w]
        cv2.namedWindow("Largest Contour", cv2.WINDOW_NORMAL)
        cv2.imshow("Largest Contour", ROI)
        cv2.waitKey(0)



def dent_detection(img, num, angle):
    accum = np.zeros_like(img)
    ksize = 9
    for theta in np.arange(0, np.pi, np.pi / 4):
        kernel = cv2.getGaborKernel((ksize, ksize), 10, theta, 18, 0.25, 0, ktype=cv2.CV_32F)
        kernel /= 1.5 * kernel.sum()
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kernel)
        np.maximum(accum, fimg, accum)
    #
    # save_image('A2_accum', accum)
    _, thresh = cv2.threshold(accum, 161, 255, cv2.THRESH_BINARY)

    sobel = convolve_sobel(img=accum,
                           threshold=18,
                           sobel_kernel_left_right=True,
                           sobel_kernel_right_left=True,
                           sobel_kernel_top_bottom=True,
                           sobel_kernel_bottom_top=True,
                           sobel_kernel_diagonal_top_left=True,
                           sobel_kernel_diagonal_bottom_left=False,
                           sobel_kernel_diagonal_top_right=True,
                           sobel_kernel_diagonal_bottom_right=False)
    sobel = cv2.convertScaleAbs(sobel)
    #save_image("A2_sobel" + str(num), sobel)

    # pos = np.where(accum < 162)
    # test_image[pos] = 255
    # save_image("A5_test_image" + str(num), test_image)

    test_image = np.zeros_like(sobel)

    pos_y, pos_x = np.where(sobel == 255)
    if pos_x.size != 0:
        if angle > -45:
            #right teeth
            print("right teeth",angle)

            for x in np.nditer(pos_x):
                for y in np.nditer(pos_y):
                    if x < 80:
                        test_image[(y, x)] = sobel[(y, x)]
            save_image('testimage' + str(num), test_image)

        if angle < -45 and angle > -75:
            #left teeth
            print("left teeth" + str(num), angle)

            for x in np.nditer(pos_x):
                for y in np.nditer(pos_y):
                    if x > 20:
                        test_image[(y, x)] = sobel[(y, x)]
            save_image('testimage' + str(num), test_image)

        if angle < -75:
            # front teeth
            test_image = sobel
            save_image('testimage' + str(num), test_image)

    return True


if __name__ == "__main__":
    path = "C:/Users/sudha/Desktop/trendzlink/Gear_image_processing/IMAGES/OK_PART_RPI-B/ok_cam1/"
    file_list = []
    file_list = os.listdir(str(path))

    for img in file_list:
        if img.endswith(".jpg"): #2020_3_5_18_54_25
            ind +=1
            image_name = img
            print ("imagename", image_name)

            
            original_image = cv2.imread(path+img)
            save_image("A0_original", original_image)

            start_time = time.time()

            l = cv2.cvtColor(original_image,cv2.COLOR_BGR2LAB)[0]
            l_mean = np.mean(l)
            print("l mean",l_mean)
            

            teeth_seperation_method1(original_image)
            # teeth_seperation_new(original_image)

            end_time = time.time()
            print ("time taken:", end_time - start_time)
            break


