import cv2,math,numpy as np,time,os,matplotlib.pyplot as plt
from envision import crop
from envision.convolution import convolve_sobel
from linear_reg import linear_reg_sc
count,local_index = 0,0

def save_image(name,image):
    global count,local_index
    cv2.imwrite("C:/Users/sudha/Desktop/trendzlink/Gear_image_processing/sudhanshu/analysis/"+str(count)+"_"+str(local_index)+"_"+str(name) +".jpg", image)
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
    
    green_channel = cv2.medianBlur(green_channel,21)
    save_image("A2_green_blur",green_channel)
    
    img1, _ = crop.crop_radial_arc_two_centres(green_channel, centre_x1=400, centre_y1=-240,
                                                 centre_x2=400, centre_y2=-300, radius1=500, radius2=650,
                                                 theta1=230,
                                                 theta2=310)
    img1_bk = img1.copy()
    save_image("A3_crop",img1)  
                                                
    img1[np.where(img1 < 200)] = [0]
    save_image("A4_binary",img1)

    time2 = time.time()
    find_teeth_freq(img1_bk)
    #find_teeth(img1,img1_bk)
    print("time taken for freq:",abs(time.time()-time2))
    return
    
def process_image(block_img,num):
    print("num:",num)
    new_pos_y,y1,x1 = [],[],[]
    block_img_bk = block_img.copy()
    blank_img = np.zeros_like(block_img_bk)

    pos_y, pos_x = np.where(block_img_bk>=200)
    # coordinates = list(zip(pos_y,pos_x))
    # new_ = np.asarray([(y,x) for (y,x) in coordinates if np.count_nonzero(pos_x==x) > 40 and np.count_nonzero(pos_y==y) > 15])
    # for y,x in zip(pos_y,pos_x):
    #     if np.count_nonzero(pos_x==x) > 10 and np.count_nonzero(pos_y==y) > 15 and np.count_nonzero(pos_y==y) < 50:
    #         y1.append(y)
    #         x1.append(x)
    #         blank_img[(y,x)] = block_img_bk[(y,x)]
    # save_image("A7_new_coord",blank_img)

    new_pos_y = linear_reg_sc(pos_y,pos_x)
    # # return
    # for x in pos_x:
    #     new_pos_y.append(int(b1*x+b2))

    coordinates = list(zip(new_pos_y,pos_x))
    print(coordinates)

    block_img = cv2.merge((block_img,block_img,block_img))

    for (y,x) in coordinates:
       block_img[(int(y),x)] = [0,0,255]

    save_image("A7_line",block_img)

    block_img_bk[np.where(block_img_bk < 200)] = [0]
    save_image("A5_block_"+str(num),block_img_bk)
    
    return
    
    
    
    
    
    
    coordinates = list(zip(pos_y,pos_x))
    print("coordinates:",coordinates)
    
    new_coordinates = np.asarray([(y,x) for (y,x) in coordinates if np.count_nonzero(pos_x==x) > 40 and np.count_nonzero(pos_y==y) > 15])

    #print("new_coordinates:",new_coordinates)
    
    #show_image("A5_block_"+str(num),block_img_bk)
    
    return

    for (y,x) in new_coordinates:
       blank_img[(y,x)] = block_img_bk[(y,x)]
    save_image("A6_teeth_manip_"+str(num),blank_img)
    
    return
    
    """
    freq_x= np.asarray(np.unique(pos_x,return_counts=True)).T
    freq_y= np.asarray(np.unique(pos_x,return_counts=True)).T
    print("freq_x:",freq_x)
    print("freq_y:",freq_y)
    
    freq = np.asarray([(x_val,count) for (x_val,count) in freq_x if count>60])
    """
    print("freq_x :",freq)
    return

def find_teeth_freq(img1):
    img1_bk = img1.copy()
    
    blank_image = np.zeros_like(img1)
    y1,x1,rev = 100, 0, False
    
    num = 0

    for (x2) in range(150,img1.shape[1],100):
        num+=1
        block_img = []
        y2 = y1 + 250
        
        block_img = img1_bk[y1:y2,x1:x2]
        
        process_image(block_img, num)

        # break
        
        blank_image[y1:y2,x1:x2] = img1[y1:y2,x1:x2]
        # blank_image[y1:y1+1,x1:x2] = 255
        # blank_image[y1:y2,x1:x1+1] = 255
        # blank_image[y2-1:y2,x1:x2] = 255
        # blank_image[y1:y2,x2-1:x2] = 255

        x1 = x2 - 50
        y1 = y1 +25 if not rev else y1-25
        
        if y1>150:
            rev = True
    
    # save_image("A5_blank",blank_image)
    return
    
if __name__ == "__main__":
    
    path = "C:/Users/sudha/Desktop/trendzlink/Gear_image_processing/IMAGES/OK_PART_RPI-B/ok_cam1/"
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

