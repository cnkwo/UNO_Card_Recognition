import cv2
import numpy as np

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
        print(image)

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

def getCardColour(self, imgRotated):
        
        # Assigning ranges for all setting ranges for all possible card colors in hsv
        redLower = (232,57,28)
        redUpper = (255,81,52)

        greenLower = (20,51,43)
        greenUpper = (65,255,71)

        blueLower = (9,113,234)
        blueUpper = (39,145,255)

        yellowLower = (210,162,40)
        yellowUpper = (255,200,70)
        
        r,g,b = self.getDominantColour(imgRotated)

        if (r in range(redLower[0], redUpper[0])) and (g in range(redLower[1], redUpper[1])) and (b in range(redLower[2], redUpper[2])):
            return "red"
        elif (r in range(greenLower[0], greenUpper[0])) and (g in range(greenLower[1], greenUpper[1])) and (b in range(greenLower[2], greenUpper[2])):
            return "green"
        elif (r in range(blueLower[0], blueUpper[0])) and (g in range(blueLower[1], blueUpper[1])) and (b in range(blueLower[2], blueUpper[2])):
            return "blue"
        elif (r in range(yellowLower[0], yellowUpper[0])) and (g in range(yellowLower[1], yellowUpper[1])) and (b in range(yellowLower[2], yellowUpper[2])):
            return "yellow"
        else:
            return "black"