import cv2,os, numpy as np,time,math
from envision import crop
from envision.convolution import convolve_sobel
count,local_index,threshold_value = 0,0,0

def save_image(name,image):
    global count,local_index
    cv2.imwrite("C:/Users/sudha/Desktop/trendzlink/Gear_image_processing/sudhanshu/analysis/"+str(count)+"_"+str(local_index)+"_"+str(name) +".jpg", image)
    local_index+=1

def drawBorder(image,x0,y0,x1,y1):
    image1 = cv2.merge((image,image,image))

    image1[y0:y0+2,x0:x1] = [0,0,255]
    image1[y0:y1,x0:x0+2] = [0,0,255]
    image1[y1-2:y1,x0:x1] = [0,0,255]
    image1[y0:y1,x1-2:x1] = [0,0,255]
        
    save_image("A5_one_teeth",image1)
    return

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

def find_contour(image):
    ch_all = cv2.merge((image,image,image))
    num = 0
    contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    if contours:
        for c in contours:
            area = cv2.contourArea(c)
            if area < 2000 or area > 100000:
                continue
            num += 1
            print(" tooth count: " + str(num) + "\t area: " + str(area))
            rect = cv2.minAreaRect(c)
            print(rect)
            width = int(rect[1][0])
            height = int(rect[1][1])
            angle = rect[2]
            print("\t\t\t\t", angle)


            box = cv2.boxPoints(rect)
            box = np.int0(box)
            
            img = cv2.drawContours(ch_all,[box], -1, (0,255,0), 3)
            save_image("teeth_marked",img)
            

            M = cv2.moments(box)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = 0, 0

            image_patch = sub_image(image, (cX, cY), angle + 90, height, width)
            save_image("teeth", image_patch)
            # blue_img = image_patch[:,:,0]
            # showimage("processed_img" + str(num), blue_img)
            # dent_detection(blue_img, num, angle)

        return 'Teeth seperated successfully'

def find_teeth_freq(image):
    row,col = image.shape
    
    pos_y,pos_x = np.where(image[0:int(row/2),0:int(col/3)]>=200)
    print("pos_y,pos_x",pos_y,pos_x)
    y_min,y_max = np.min(pos_y),np.max(pos_y)
    print("y_min,y_max",y_min,y_max)
    
    coordinates = np.asarray(list(zip(pos_x,pos_y)))
    # neW = np.asarray([(y,x) for (y,x) in coordinates if np.count_nonzero(pos_x==x) > 40 and np.count_nonzero(pos_y==y) > 15])
    x0 = pos_x[np.where(pos_y<=np.min(pos_y))]
    x0 = int(np.median(x0))
    print("x0:",x0)

    image[(np.min(pos_y)+60):(np.min(pos_y)+62),0:image.shape[1]] = 0
    save_image("A7_trial",image)
    x1 = pos_x[np.where(pos_y>(np.min(pos_y)+60))]
    print("x1:",np.min(x1))

    print("coordinates:",coordinates[0])
    print("minimun x,y:",list(pos_x).index(np.min(pos_x)),np.min(pos_y))
    print("max x,y:",np.max(pos_x),np.max(pos_y))

    drawBorder(image,np.min(pos_x),np.min(pos_y),np.max(pos_x),np.max(pos_y))
    

def teeth_seperation(image):
    global threshold_value

    threshold_value = 180

    color_img = image.copy()

    green_channel = color_img[:,:,1]
    save_image("A2_green_a",green_channel)

    image_bk = green_channel.copy()
    green_channel = cv2.medianBlur(green_channel,21)
    save_image("A2_green_blur",green_channel)
    
    img1, _ = crop.crop_radial_arc_two_centres(green_channel, centre_x1=400, centre_y1=-270,
                                                 centre_x2=400, centre_y2=-280, radius1=500, radius2=650,
                                                 theta1=230,
                                                 theta2=310)

    img1_bk = img1.copy()
    save_image("A3_crop",img1)  
                                                
    image_bk[np.where(img1 <= threshold_value)] = [0]
    save_image("A4_binary",image_bk)

    find_contour(image_bk)
    time2 = time.time()
    # find_teeth_freq(img1_bk)
    #find_teeth(img1,img1_bk)
    print("time taken for freq:",abs(time.time()-time2))

    return


if __name__ == "__main__":
    
    path = "C:/Users/sudha/Desktop/trendzlink/Gear_image_processing/IMAGES/OK_PART_RPI-B/ok_cam1/"
    path = "C:/Users/sudha/Desktop/trendzlink/Gear_image_processing/sudhanshu/"
    
    image_list = []
    files_list = os.listdir(str(path))
    original_image = np.zeros((600,800,3),dtype = 'uint8')

    for files in files_list:
        if files.endswith(".jpg"):
            print("image name:",files)
            count+=1
            original_image = (cv2.imread(path+files))
            #print(original_image.shape)
            #save_image("A0_original",original_image)
            
            time1 = time.time()
            teeth_seperation(original_image)
            print("time taken for whole process",abs(time.time()-time1))
            
            break
    # flank_seperation(image)

