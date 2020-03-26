import cv2,math,numpy as np,time,os,matplotlib.pyplot as plt
from envision import crop
from envision.convolution import convolve_sobel
from linear_reg import linear_reg_sc
count,local_index = 0,0

def save_image(name,image):
    global count,local_index
    cv2.imwrite("output/"+str(count)+"_"+str(local_index)+"_"+str(name) +".jpg", image)
    local_index+=1

def show_image(name,image):
    global count
    cv2.imshow(str(count)+"_"+str(name) +".jpg", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def teeth_seperation(image):
    
    num = 0

    color_img = image.copy()
    #gray_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    #save_image("A1_gray",gray_img)
    
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
                                                
    image_bk[np.where(img1 < 200)] = [0]
    save_image("A4_binary",image_bk)

    time2 = time.time()

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

