import os 
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
import joblib
from math import trunc

imageFiles = []
desImageFiles = []

IMAGES_FOLDER_PATH = "images"
DES_IMAGES_PATH = "trainImages"

for filename in os.listdir(IMAGES_FOLDER_PATH):
    if filename.endswith('.jpg'):
        imageFiles.append(cv2.imread(f'{IMAGES_FOLDER_PATH}/{filename}'))
for filename in os.listdir(DES_IMAGES_PATH):
    desImageFiles.append(cv2.imread(f'{DES_IMAGES_PATH}/{filename}'))

# Reading an image
img = cv2.imread('images/y1.jpg') # Imports the image from image folder

# Converting an image to grayscale
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Thresholding an image
ret, imgThresh = cv2.threshold(imgGray, 180, 255, cv2.THRESH_BINARY)

# Creating a kernal which uses a matrix of 5
kernel = np.ones((5,5), np.uint8)

# Removing the noise from the image using morphology
imgMorph = cv2.morphologyEx(imgThresh, cv2.MORPH_CLOSE, kernel)
imgMorph = cv2.morphologyEx(imgThresh, cv2.MORPH_OPEN, kernel)

# Copying the image to display contour on
imgCopy = img.copy()
imgCopy2 = img.copy()

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

# Extracting all contours on binarised image
contours, hierarchy = cv2.findContours(imgThresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

imgInternalContour = cv2.drawContours(imgCopy2, contours, trunc(len(contours)/2)-1, (0,255,0), 3)
print(trunc(len(contours)/2))

imgRotated = cv2.rotate(imgCropped, cv2.ROTATE_90_COUNTERCLOCKWISE)

def threshold(img):
    '''
    This method recieves an openCV image (img) as an argument and returns a binarised (threshold) image.
    params: 
        img: openCV image object.
    returns: 
        imgThresh: Binarised openCV object.
    '''        
    # Converting cv image to grayscale
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Thresholding the image (Binarisation)
    ret, imgThresh = cv2.threshold(imgGray, 180, 255, cv2.THRESH_BINARY)

    return imgThresh

def morphology(img):
    '''
    This method recieves a binarised image and removes image noise by using morphology
    params: 
        img: Binarised openCV object
    returns: 
        imgMorph: 
    '''
    # Creating a kernal which uses a matrix of 5
    kernel = np.ones((5,5), np.uint8)

    # Removing the noise from the image using morphology (morph open & close)
    imgMorph = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    imgMorph = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    return imgMorph

def edgeDetection(img):
    '''
    This method recieves an openCV image which has been morphologized as an argument
    and returns a dictionary containing the vertices for image cropping
    params:
        img: morphologized openCV image
    returns:
        croppingCoordinates: cropping coordinates dictionary
    '''

    # Extracting the external (perimeter) contour on the image
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Creating cropping vertices dictionary
    croppingCoordinates = {}
    for c in contours:
        [x,y,w,h] = cv2.boundingRect(c)
        croppingCoordinates = {'x':x,'y':y,'w':w,'h':h}

    return croppingCoordinates

def cropImage(croppingCoordinates, img):
    '''
    This method recieves cropping vertices and a openCV image as arguments and returns a cropped image of just the card.
    params:
        img: openCV image object.
        croppingCoordinates: cropping coordinates dictionary
    returns:
        imgCropped: cropped openCV image object.
    '''
    # Getting dictionary values based on their key names and assigning them to variables
    x,y,w,h = croppingCoordinates['x'], croppingCoordinates['y'], croppingCoordinates['w'], croppingCoordinates['h']

    # Cropping image based on values
    imgCropped = img[y:y+h, x:x+w]

    return imgCropped

def rotateImage(img):
    '''
    This method recieves a cropped openCV image and returns a rotated version the image
    params:
        img: cropped openCV image object.
    returns:
        imgRotated: rotated openCV image object.
    '''
    # Rotates image, and assigning it to a variable
    imgRotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    
    return imgRotated

def getCardColour(imgRotated):
        
    def createHistogram(cluster):
        '''
        This method creates a histogram using the k clusters data values
        param: clusters
        return: hist
        '''

        numLabels = np.arange(0, len(np.unique(cluster.labels_)) + 1)
        hist, _ = np.histogram(cluster.labels_, bins=numLabels)
        hist = hist.astype('float32')
        hist /= hist.sum()

        return hist

    def plotColours(hist, centroids):

        bar = np.zeros((50, 300, 3), dtype="uint8")
        start = 0

        for (percent, color) in zip(hist, centroids):
            
            # plot the relative percentage of each cluster
            end = start + (percent * 300)
            cv2.rectangle(bar, (int(start), 0), (int(end), 50), color.astype("uint8").tolist(), -1)
            start = end

        # Getting the dominant color's index number
        dominantColorIndex = np.where(hist==max(hist))
        
        # Converting the index values into integers for BGR arrangement
        b,g,r = int(centroids[dominantColorIndex][0][0]), int(centroids[dominantColorIndex][0][1]), int(centroids[dominantColorIndex][0][2])
        
        # Assigning BGR arrangment as the dominant color tuple 
        dominantColor = (b,g,r)

        return bar, dominantColor

    # Getting the shape of image array (rows and columns)
    height, width, _ = np.shape(imgRotated)

    image = imgRotated.reshape((height * width, 3))

    # Clusters number
    num_clusters = 3
    clusters = KMeans(n_clusters=num_clusters)

    try:
        clusters.fit(image)
        hist = createHistogram(clusters)
        colourBar, colourBGR = plotColours(hist, clusters.cluster_centers_)
        return(colourBar, colourBGR) 

    except Exception as e:
        return "N/A"

# Empty x_train list
x_train = []

# Creating y train data - each card is classified by its color [0 = blue, 1 = green, 2 = black, 3 = red, 4 = yellow]
y_train = [0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,3,3,3,3,3,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4,4,4,4,4]

# Append each cards b,g,r tuples into x train data list
for img in imageFiles:

    # Getting threshold image
    imgThresh = threshold(img)

    # Using morphology to remove the image noise
    imgMorph = morphology(imgThresh)
    
    # Getting the card (perimeter) contour
    croppingCoordinates = edgeDetection(imgMorph)

    # Cropping the image to just the card by using the card contour
    imgCropped = cropImage(croppingCoordinates, img)

    # Getting the bgr tuple
    cardColourBar, cardBGR = getCardColour(imgCropped)

    x_train.append(cardBGR)

# Creates Classifier
clf = KNeighborsClassifier(2)

clf.fit(x_train, y_train)

modelName = "c_classifer.sav"

joblib.dump(clf, modelName)

print(x_train)
print("Model has been saved")

myModel = joblib.load(modelName)

colourBar, colourBGR  = getCardColour(imgRotated)

result = myModel.predict([colourBGR])

print(result[0])