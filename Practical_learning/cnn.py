import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
#from keras.preprocessing.image 


path = "trainingData"
testRatio = 0.2
valRatio = 0.2

count = 0
images = []
classNo = []
myList = os.listdir(path)

print("Total No of Classes Detected: ", len(myList))

noOfClasses = len(myList)

print("Importing Classes.....")
for x in range(0,noOfClasses):
    myPicList = os.listdir(path + "/" + str(count))
    print(myPicList)
    for y in myPicList:
        #print(y)
        curImg = cv2.imread(path+"/"+str(count)+"/"+y)
        #curImg = cv2.cvtColor(curImg, cv2.COLOR_BGR2GRAY)
        curImg = cv2.resize(curImg,(640,480))
        images.append(curImg)
        classNo.append(count)

    print(count, end=" ")
    count += 1
print(" ")

print("The number of images in the images list = ", len(images))
print("The number of images in the classNo list = ", len(classNo))

images = np.array(images)
classNo = np.array(classNo)

print(images.shape)
#print(classNo.shape)

### splitting the data
x_train, x_test, y_train, y_test = train_test_split(images, classNo, test_size=testRatio)
x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=valRatio)

print(y_train.shape, "vv")
print(x_test.shape)
print(x_validation.shape)


noOfSamples = []
print(len(np.where(x_train==1)))

for x in range(0, noOfClasses):
    #print(len(np.where(x_train==x)))
    noOfSamples.append(len(np.where(x_train==x)))

print(noOfSamples)

plt.figure(figsize=(10,5))
plt.bar(range(0,noOfClasses), noOfSamples)
plt.title("Number of images for each class")
plt.xlabel("Class ID")
plt.ylabel("Number of Images")
plt.show()

print(x_train[3].shape, "ff")

def preProcessing(img):
    img = cv2.cvtColor(curImg, cv2.COLOR_BGR2GRAY)
    
    # Thresholding an image
    ret, imgThresh = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5,5), np.uint8)

    # Removing the noise from the image using morphology
    imgMorph = cv2.morphologyEx(imgThresh, cv2.MORPH_CLOSE, kernel)
    imgMorph = cv2.morphologyEx(imgThresh, cv2.MORPH_OPEN, kernel)

    # Copying the image to display contour on
    imgCopy = img.copy()

    # Extracting external (perimeter) contour on binarised image
    contours, hierarchy = cv2.findContours(imgMorph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Drawing the contour onto the image copy
    imgContoured = cv2.drawContours(imgCopy, contours, -1, (0, 255, 0), 2)

    croppingCoordinates = {}
    for c in contours:
        [x,y,w,h] = cv2.boundingRect(c)
        croppingCoordinates = {'x':x,'y':y,'w':w,'h':h}

    print(croppingCoordinates)

    x,y,w,h = croppingCoordinates['x'], croppingCoordinates['y'], croppingCoordinates['w'], croppingCoordinates['h']

    imgCropped = img[y:y+h, x:x+w]
    
    imgRotated = cv2.rotate(imgCropped, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # Equalize image
    img = cv2.equalizeHist(img)

    # Normalize values
    img = imgRotated/255

    return img

#img = preProcessing(x_train[1][2])
#img = cv2.resize(img,(640,480))
#cv2.imshow("PreProcessed", img)
#cv2.waitKey(0)
#print(x_train)


x_train = np.array(list(map(preProcessing, x_train)))
x_test = np.array(list(map(preProcessing, x_test)))
x_validation = np.array(list(map(preProcessing, x_validation)))

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1)
x_validation = x_validation.reshape(x_validation.shape[0],x_validation.shape[1],x_validation.shape[2],1)


img = x_train[3]
#img = cv2.resize(img,(640,480))
cv2.imshow("PreProcessed", img)
cv2.waitKey(0)