# Saylee Kanitkar
#code to find elipse in an image
# this code is not useful
import cv2
import numpy as np

# Load image
# image = cv2.imread('/media/saylee/Work/Allygrow/Gear_image_processing/IMAGES/OK_PART_RPI-B/ok_cam1/2020_3_18_18_51_56.jpg', 0)
# image = cv2.imread('/home/saylee/Downloads/blue_new.png', 0)
image = cv2.imread('/home/saylee/Downloads/sobel.jpeg', 0)
# image = cv2.imread('/home/saylee/Downloads/2020_3_18_18_46_10.jpg', 0)

ret, thresh = cv2.threshold(image, 70, 255, 0)
cv2.imshow("Binary",thresh)
cv2.waitKey(0)

# Set our filtering parameters
# Initialize parameter settiing using cv2.SimpleBlobDetector
params = cv2.SimpleBlobDetector_Params()

# Set Area filtering parameters
params.filterByArea = True
params.minArea = 100

# Set Circularity filtering parameters
# params.filterByCircularity = True
params.filterByCircularity = False

# params.minCircularity = 0.9
# params.minCircularity = 0.1


# Set Convexity filtering parameters
params.filterByConvexity = True
# params.filterByConvexity = False
params.minConvexity = 0.2

# Set inertia filtering parameters
# params.filterByInertia = True
params.filterByInertia = False
params.minInertiaRatio = 0.01

# Create a detector with the parameters
detector = cv2.SimpleBlobDetector_create(params)

# Detect blobs
keypoints = detector.detect(image)

# Draw blobs on our image as red circles
blank = np.zeros((1, 1))
blobs = cv2.drawKeypoints(image, keypoints, blank, (0, 0, 255),
                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

number_of_blobs = len(keypoints)
text = "Number of Circular Blobs: " + str(len(keypoints))
cv2.putText(blobs, text, (20, 550),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2)

# Show blobs
cv2.imshow("Filtering Circular Blobs Only", blobs)
cv2.waitKey(0)
cv2.destroyAllWindows()
