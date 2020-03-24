import cv2,os, numpy as np,time
from envision import crop
from envision.convolution import convolve_sobel
count,local_index,threshold_value = 0,0,0

def save_image(name,image):
    global count,local_index
    cv2.imwrite("C:/Users/sudha/Desktop/trendzlink/Gear_image_processing/sudhanshu/analysis/"+str(count)+"_"+str(local_index)+"_"+str(name) +".jpg", image)
    local_index+=1

def drawBorder(image,x0,y0,x1,y1):
    image[y0:y0+2,x0:x1] = [0,0,255]
    image[y0:y1,x0:x0+2] = [0,0,255]
    image[y1-2:y1,x0:x1] = [0,0,255]
    image[y0:y1,x1-2:x1] = [0,0,255]
    return image

def find_teeth_freq(image):
    row,col = image.shape
    pos_y,pos_x = np.where(image[0:int(row/2),0:int(col/3)]>=200)
    print("pos_y,pos_x",pos_y,pos_x)
    coordinates = list(zip(pos_x,pos_y))
    print("coordinates:",coordinates)
    print("minimun x,y:",list(pos_x).index(np.min(pos_x)),np.min(pos_y))
    print("max x,y:",np.max(pos_x),np.max(pos_y))

    image1 = cv2.merge((image,image,image))
    image1 = drawBorder(image1,np.min(pos_x),np.min(pos_y),np.max(pos_x),np.max(pos_y))
    # image[np.where(image[0:int(row/2),0:int(col/3)]>=200)] = 0
    save_image("A5_one_teeth",image1)

def teeth_seperation(image):
    global threshold_value

    threshold_value = 200

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
                                                
    image_bk[np.where(img1 <= 200)] = [0]
    save_image("A4_binary",image_bk)

    color_img[np.where(img1 <= 200)] = [0,0,0]
    save_image("A4_binary",color_img[:,:,2])

    time2 = time.time()
    find_teeth_freq(img1_bk)
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

