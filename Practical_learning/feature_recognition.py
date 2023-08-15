import cv2
import numpy as np

# Convert images to grayscale to make it easier to recognise
img1 = cv2.imread("images/b0.jpg", 0)
img2 = cv2.imread("trainImages/b0.jpg", 0)

orb = cv2.ORB_create()

# Helps find key points and descriptors for images
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img1,None)

# Show Key points
imgKp1 = cv2.drawKeypoints(img1,kp1,None)
imgKp2 = cv2.drawKeypoints(img2,kp2,None)

# The descriptors are basically an array of numbers which help describe the key points
#print(des1)

# Here we can view the shape
# Basically the orb detector uses 500 points. Therefore it will try to find 500 features in each image. 
#print(des1.shape)
# And will describe them using 32 values
#print(des1[0])


# Get matches
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

# Check if matches are within a certain distance between each feature on images and train image. If so, add them to a list of good matches
good = []
for m, n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])

# Shows how many good matches we have
print(len(good))

# Image showing good matches between the two images
img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)

# Show images with key points 
cv2.imshow('Kp1', imgKp1)
cv2.imshow('Kp2', imgKp2)

# Show images (both normal and train images)
cv2.imshow('img1', img1)
cv2.imshow('img2', img2)

# Shows where the appropriate matches are
cv2.imshow('img3', img3)

cv2.waitKey(0)

