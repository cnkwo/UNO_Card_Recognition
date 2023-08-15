import cv2
import numpy as np
print('package imported')

# Basic functions for OpenCV

img = cv2.imread('images/b0.jpg') # Imports the image from image folder
kernel = np.ones((5,5), np.uint8)

# Converting an image to grayscale
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Blurring an image
imgBlur = cv2.GaussianBlur(img,(7,7),0)

# Edge detection
imgCanny = cv2.Canny(img,100,100)

# Adjusting edge thickness
imgDialation = cv2.dilate(imgCanny, kernel, iterations=1)

# Erosion of edge thickness
imgEroded = cv2.erode(imgDialation, kernel, iterations=1)

cv2.imshow('Image output', img) # To display normal image
cv2.imshow('Gray Image output', imgGray) # To display gray image
cv2.imshow('Blurred Image output', imgBlur) # To display blurred image
cv2.imshow('Canny Image output', imgCanny) # To display image with edges
cv2.imshow('Dialation Image output', imgDialation) # To display image with thicker edges
cv2.imshow('Eroded Image output', imgEroded) # To display image with thiner edges


cv2.waitKey(0)
