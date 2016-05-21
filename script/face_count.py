# face_count - created by Akilesh Praveen
# May 20
# www.github.com/akipraveen

import cv2

# file paths
path_to_img = 'resources/facial_profiles/1.jpg'
path_to_casc = 'resources/classifiers/haarcascade_frontalface_default.xml'

faceCascade = cv2.CascadeClassifier(path_to_casc)

print path_to_img

# Reading the image file
img = cv2.imread(path_to_img)

# Create a gray version of the Image
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detecting the faces
faceArray = faceCascade.detectMultiScale(
    img_gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
    flags = cv2.CASCADE_SCALE_IMAGE
)

print "# of faces found: ", len(faceArray)