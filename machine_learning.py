''' Uno Card Recognition by Patrick Nkwocha'''

import numpy as np
import cv2
import os
import joblib
from image_processing import ImageProcessing
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
#import utils


class MachineLearning():

    def __init__(self):
        self.ip = ImageProcessing()
        self.cardNames = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'Reverse', 'Stop','Draw 2', '7', 'Blank', 'Swap Hands', 'Draw 4', 'Wild', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'Reverse', 'Stop','Draw 2']
        self.textCards = ['Reverse', 'Stop','Draw 2', 'Blank', 'Swap Hands', 'Draw 4', 'Wild']
        self.modelName = "c_classifer.sav"
        self.testPath = "testImages"
        self.imgHeight = 640
        self.imgWidth  = 480

    def findID(self, imgMark, desList, thres=7):

        orb = cv2.ORB_create(nfeatures=1000)

        # Find key points and descriptors for images
        kp2,des2 = orb.detectAndCompute(imgMark,None)
        
        bf = cv2.BFMatcher()
        matchList = []
        finalVal = -1

        try:
            for des in desList:

                # Get matches
                matches = bf.knnMatch(des,des2, k=2)
                
                # Check if matches are within a certain distance between each feature on images and train image. If so, add them to a list of good matches
                good = []
                for m, n in matches:
                    if m.distance < 0.75 * n.distance:
                        good.append([m])

                matchList.append(len(good))
        except:
            pass
        
        #print(matchList)
        # Check if matches were found and if total maximum num of matches exceeds the threshold
        if len(matchList) != 0 and max(matchList) > thres:
            # index of image with max matches is set to
            finalVal = matchList.index(max(matchList))
        else:
            finalVal == -1

        return finalVal

    def getCardColour(self, imgRotated):
        
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

            for (percent, colour) in zip(hist, centroids):
                
                # Plotting the percaentage of each cluster
                end = start + (percent * 300)
                cv2.rectangle(bar, (int(start), 0), (int(end), 50), colour.astype("uint8").tolist(), -1)
                start = end

            # Getting the dominant color's index number
            dominantColourIndex = np.where(hist==max(hist))
            
            # Converting the index values into integers for BGR arrangement
            b,g,r = int(centroids[dominantColourIndex][0][0]), int(centroids[dominantColourIndex][0][1]), int(centroids[dominantColourIndex][0][2])
            
            # Assigning BGR arrangment as the dominant color tuple 
            dominantColour = (b,g,r)

            return bar, dominantColour

        # Getting the shape of image array (rows and columns) -- Image attributes
        height, width, _  = np.shape(imgRotated)

        image = imgRotated.reshape((height * width, 3))

        # Clusters number
        num_clusters = 5
        clusters = KMeans(n_clusters=num_clusters)

        ######
        try:
            clusters.fit(image)
            hist = createHistogram(clusters)
            colourBar, colourBGR = plotColours(hist, clusters.cluster_centers_) 
            return(colourBar, colourBGR)

        except Exception as e:
            return "N/A"
    
    def getColourName(self, colourBGR):
        '''
        This method finds the colour of the card by using KNeighborsClassifier. This done by creating a training dataset
        To which the colourBGR is placed amongst and where its neareset neighbour is assumed its class.
        params: colourBGR
        return: colourName
        '''
        
        def createClassifyingData():
            '''
            This method is responsible for writing the training dataset
            '''
            # Empty x_train list
            x_train = []

            # Creating y train data - each card is classified by its colour [0 = blue, 1 = green, 2 = black, 3 = red, 4 = yellow]
            y_train = [0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,3,3,3,3,3,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4,4,4,4,4]

            # Appending each cards b,g,r tuple into x_train data list
            for img in self.ip.imageFiles:

                # Getting threshold image
                imgThresh = self.threshold(img)

                # Using morphology to remove the image noise
                imgMorph = self.ip.morphology(imgThresh)
                
                # Getting the card (perimeter) contour
                croppingCoordinates, contours = self.ip.edgeDetection(imgMorph)

                # Cropping the image to just the card by using the card contour
                imgCropped = self.ip.cropImage(croppingCoordinates, img)

                # Rotating the cropped image 
                imgRotated = self.ip.rotateImage(imgCropped)

                # Get cards BGR values
                cardColourBar, cardBGR = self.getCardColour(imgRotated)

                x_train.append(cardBGR)

            # Create Classifier
            clf = KNeighborsClassifier(2)
            clf.fit(x_train, y_train)

            # Saving the data model
            joblib.dump(clf, self.modelName)
        
        # Loading the saved model 
        myModel = joblib.load(self.modelName)
        
        # Model result
        result = myModel.predict([colourBGR])
        
        # Result query
        if result[0] == 0:
            return 'Blue'
        elif result[0] == 1:
            return 'Green'
        elif result[0] == 2:
            return 'Black'
        elif result[0] == 3:
            return 'Red'
        elif result[0] == 4:
            return 'Yellow'
        else:
            return 'N/A'

    def startStream(self):
        
        cameraStream = cv2.VideoCapture(0)

        desList = self.ip.findDes()

        while True:
            
            success, cameraFrame = cameraStream.read()
            img = cameraFrame.copy()
            testImg = cameraFrame.copy()
            cameraFrame = cv2.cvtColor(cameraFrame,cv2.COLOR_BGR2GRAY)

            cv2.putText(img, '- Enter key to observe card', (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
            cv2.putText(img, '- Space key to exit stream', (50,80), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)

            cv2.imshow('Camera Stream', img)

            img = cv2.resize(img, (self.imgWidth, self.imgHeight))

            key = cv2.waitKey(1)

            if (key == 32):

                cv2.destroyAllWindows()
                cameraStream.release()
                break


            if (key == 13):

                # Save captured frame (image)
                filename = 'savedImage.jpg'
                cv2.imwrite(self.testPath + "/" + filename, testImg)

                savedImg = cv2.imread('testImages/'+filename)

                # Getting threshold image (binarisation)
                imgThresh = self.ip.threshold(savedImg)

                # Using morphology to remove the image noise on binarised image
                imgMorph = self.ip.morphology(imgThresh)
                
                # Getting the card (perimeter) contour
                croppingCoordinates, contours = self.ip.edgeDetection(imgMorph)

                # Extracting the external (perimeter) contour on the image
                #contours, hierarchy = cv2.findContours(savedImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

                # Cropping the image to just the card by using the card contour
                imgCropped = self.ip.cropImage(croppingCoordinates, savedImg)

                biggest, maxArea = self.ip.findBiggestContour(contours)

                if biggest.size != 0:

                    biggest = self.ip.reorderImage(biggest)
            
                    # Assign warping points to straighten out image (remove curves)
                    pts1 = np.float32(biggest) 
                    pts2 = np.float32([[0, 0],[self.imgWidth, 0], [0, self.imgHeight],[self.imgWidth, self.imgHeight]])
                    matrix = cv2.getPerspectiveTransform(pts1, pts2)
                    imgWarp = cv2.warpPerspective(img, matrix, (self.imgWidth, self.imgHeight))

                    # Remove pixels from each corner of the image
                    imgWarp = imgWarp[20:imgWarp.shape[0] - 20, 20:imgWarp.shape[1] - 20]
                    imgWarp = cv2.resize(imgWarp,(self.imgWidth,self.imgHeight))

                    # Update image captured 
                    cv2.imwrite(self.testPath + "/" + filename, imgWarp)
                    
                    savedImg = cv2.imread('testImages/'+filename)

                # Get the cards BGR tuple
                colourBar, colourBGR = self.getCardColour(savedImg)

                # Get Card colour name using KNN classifier
                colourName = self.getColourName(colourBGR)

                id = self.findID(savedImg, desList)

                if id != -1:
                    card = self.cardNames[id]
                    if card in self.textCards:
                        if str(card) == "Wild":
                            # Alternative method to retrieve colour for wild (difficult card)
                            colourName = self.getAltCardColour(savedImg)
                            message = "This is a {} {} Uno card!!".format(colourName, card)
                        else:
                            message = "This is a {} {} Uno card!".format(colourName, card)
                    else:
                        message = "This is a {} number {} Uno card!".format(colourName, card)

                elif colourName == "N/A":
                    message = "The colour of this card has not been recognised, please try again."
                else:
                    message = "Card not regocnised please try again."

                print(message)

                cv2.putText(img, message, (150,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)    

    def getAltCardColour(self, imgRotated):
        '''
        This method recieves an image before triggering an event to retrieve its RGB values. After which it checks whether they fall within a predetermined range to return a colour name
        params: 
            imgRotated: rotated openCV image object
        returns:
            colorName:  name of cards color
        '''
        
        # Assigning ranges for all setting ranges for all possible card colors in hsv
        redLower = (232,57,28)
        redUpper = (255,81,52)

        greenLower = (20,51,43)
        greenUpper = (65,255,71)

        blueLower = (9,113,234)
        blueUpper = (39,145,255)

        yellowLower = (210,162,40)
        yellowUpper = (255,200,70)
        
        # Get the most dominant colour found in the image
        r,g,b = self.getDominantColour(imgRotated)

        if (r in range(redLower[0], redUpper[0])) and (g in range(redLower[1], redUpper[1])) and (b in range(redLower[2], redUpper[2])):
            return "Red"
        elif (r in range(greenLower[0], greenUpper[0])) and (g in range(greenLower[1], greenUpper[1])) and (b in range(greenLower[2], greenUpper[2])):
            return "Green"
        elif (r in range(blueLower[0], blueUpper[0])) and (g in range(blueLower[1], blueUpper[1])) and (b in range(blueLower[2], blueUpper[2])):
            return "Blue"
        elif (r in range(yellowLower[0], yellowUpper[0])) and (g in range(yellowLower[1], yellowUpper[1])) and (b in range(yellowLower[2], yellowUpper[2])):
            return "Yellow"
        else:
            return "Black"

    def getDominantColour(self, imgRotated):
        '''
        This method finds the dominant color of the card by using KMeans clusters.
        After which a tuple consisting of the RGB values of the dominant color is returned.
        params: imgRotated
        return: dominantColor
        '''
        # Converting the card from BGR to RGB
        imgRotated = cv2.cvtColor(imgRotated, cv2.COLOR_BGR2RGB)

        # Getting the shape of the image array (rows and columns)
        height, width, _ = np.shape(imgRotated)

        # Reshaping the image to a list consisting of rgb pixels
        image = np.float32(imgRotated.reshape(height * width, 3))

        # Criteria for KMeans cluster
        clusterCriteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1.0)

        num_clusters = 3
        attempts = 100
        
        ret, label, center = cv2.kmeans(image,num_clusters,None,clusterCriteria,attempts,cv2.KMEANS_PP_CENTERS)

        # Converts the center point of data back into 8 bit datatype (uint8)
        center = np.uint8(center)

        # For loop for checking rgb values 
        for r,g,b in center:
            if (r < 60) and (g < 60) and (b < 60):
                pass
            elif (r > 190) and (g > 190) and (b > 190):
                pass
            else:
                dominantColor = (r,g,b)
                return dominantColor