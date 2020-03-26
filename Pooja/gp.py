import os,cv2,numpy as np,time
from gear_process import gear_process
if __name__ == "__main__":
    path = "C:/Users/sudha/Desktop/trendzlink/Gear_image_processing/IMAGES/OK_PART_RPI-B/ok_cam1/"
    file_list = []
    file_list = os.listdir(str(path))

    for img in file_list:
        if img.endswith("2020_3_18_18_52_47.jpg"): #2020_3_5_18_54_25
            image_name = img
            white_pixel_list = []
            print ("imagename", image_name)
            start_time = time.time()
            original_image = cv2.imread(path+img)
            
            #object the class with the image to be processed upon
            
            gp = gear_process(original_image)


            """
            class named gear_process is created to store all the separation code for the image

            only function to access is process_separation which returns the list of images for flanks and teethts
            if the contour = False, without separating the parts, it returns 2 whole image list which highlights the flanks and teeths respectively 
            if contour = True, 2 lists with consecutive parts are returned
            """
            flank_list,teeth_list = gp.process_separation(contour = True)
            
            for teeth,cnt in zip(teeth_list,range(len(teeth_list))): cv2.imwrite("analysis/"+str(cnt)+"_teeth.jpg",teeth)
            for flank,cnt in zip(flank_list,range(len(flank_list))): cv2.imwrite("analysis/"+str(cnt)+"_flank.jpg",flank) 
            
            cv2.imwrite("analysis/A0_original.jpg",original_image)
            break
