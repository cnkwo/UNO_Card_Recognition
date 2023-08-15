''' Uno Card Recognition by Patrick Nkwocha'''

import numpy as np
import os
import cv2
from PIL import ImageTk,Image

class ImageProcessing():

    def __init__(self):
        '''Load all images'''

        self.imageFiles = []
        self.desImageFiles = []

        IMAGES_FOLDER_PATH = "images"
        DES_IMAGES_PATH = "trainImages"

        for filename in os.listdir(IMAGES_FOLDER_PATH):
            if filename.endswith('.jpg'):
                self.imageFiles.append(cv2.imread(f'{IMAGES_FOLDER_PATH}/{filename}'))
        for filename in os.listdir(DES_IMAGES_PATH):
            self.desImageFiles.append(cv2.imread(f'{DES_IMAGES_PATH}/{filename}'))

        # Get imageFiles list length
        self.listLength = len(self.imageFiles)

    def displayImage(self, imageNumber):
        '''
        This method recives an index value (imageNumber) as an argument, 
        using that index value to find its corresponding image file in the imageFiles list, 
        and converts it to a PIL image to return for display.
        params: 
            imageNumber: image index value.
        returns: 
            img: PIL image object.
        '''
        # Converting cv images BGR values to RGB
        img = cv2.cvtColor(self.imageFiles[imageNumber], cv2.COLOR_BGR2RGB)

        # Converting cv image to PIL image for GUI display
        img = ImageTk.PhotoImage(image=Image.fromarray(img))
        
        return img

    def threshold(self, img):
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

    def morphology(self, img):
        '''
        This method recieves a binarised image and removes image noise by using morphology
        params: 
            img: Binarised openCV object
        returns: 
            imgMorph: morphologized openCV object.
        '''
        # Creating a kernal which uses a matrix of 5
        kernel = np.ones((5,5), np.uint8)

        # Removing the noise from the image using morphology (morph open & close)
        imgMorph = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        imgMorph = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

        return imgMorph

    def edgeDetection(self, img):
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

        return croppingCoordinates, contours

    def cropImage(self, croppingCoordinates, img):
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

    def rotateImage(self, img):
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

    def createORB(self, img):
        '''
        This method is responsible for creating the ORB () for feature rocognition'''
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        orb = cv2.ORB_create()

        kp,des = orb.detectAndCompute(img,None)

        return kp,des

    def findDes(self):
        '''
        This method returns a list of each training images keypoints and description for ORB...
        returns:
            desList: description list for training images
        '''

        # Creating description list
        desList = []

        # Appending each cards keypoints and description into description list (desList)
        for img in self.desImageFiles:

            # Getting threshold image
            #imgThresh = self.threshold(img)

            # Using morphology to remove the image noise
            #imgMorph = self.morphology(imgThresh)

            # Getting the description card (perimeter) contour
            #croppingCoordinates = self.edgeDetection(imgMorph)

            # Cropping the image to just the card by using the card contour
            #imgCropped = self.cropImage(croppingCoordinates, img)

            # Rotating the cropped image 
            #imgRotated = self.rotateImage(imgCropped)
            
            #
            kp,des = self.createORB(img)

            desList.append(des)

        return desList

    def findBiggestContour(self, contours):
        '''
        This method recieves an openCV images contours and returns the largest rectangular contour
        params:
            contours: openCV image objects contours
        returns:
            biggest, max_area: largest rectangular contour
        '''

        biggest = np.array([])
        max_area = 0 

        for i in contours:

            area = cv2.contourArea(i)

            if area > 5000:
                perimeter = cv2.arcLength(i, True)
                approx = cv2.approxPolyDP(i, 0.02 * perimeter, True)

                # Check if it has 4 edges for card rectangle
                if area > max_area and len(approx) == 4:
                    biggest = approx
                    max_area = area

        return biggest, max_area

    def reorderImage(self, imgPoints):

        imgPoints = imgPoints.reshape((4, 2))
        imgPointsNew = np.zeros((4, 1, 2), dtype=np.int32)
        add = imgPoints.sum(1)

        imgPointsNew[0] = imgPoints[np.argmin(add)]
        imgPointsNew[3] = imgPoints[np.argmax(add)]
        diff = np.diff(imgPoints, axis=1)
        imgPointsNew[1] = imgPoints[np.argmin(diff)]
        imgPointsNew[2] = imgPoints[np.argmax(diff)]

        return imgPointsNew
