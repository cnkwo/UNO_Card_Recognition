''' Uno Card Recognition by Patrick Nkwocha'''

import tkinter as tk
import cv2
from tkinter import ttk
from tkinter import filedialog
from image_processing import ImageProcessing
from machine_learning import MachineLearning

class Gui():

    def __init__(self):
        self.root = tk.Tk()
        self.ip = ImageProcessing()
        self.ml = MachineLearning()
        self.imageNumber = 0
        self.photo = self.ip.displayImage(self.imageNumber)


    def configureGui(self):
        '''
        This method is responsible for configuring the gui (user interface)
        '''
        self.root.title("Computer Vision Project - Uno Card Recognition")


        tabControl = ttk.Notebook(self.root) # Creating Notebook

        # Adding frame for each tab
        tab1 = tk.Frame(tabControl)
        tab2 = tk.Frame(tabControl)

        # Adding tabs and tab names
        tabControl.add(tab1, text ="Photo Recognition")
        tabControl.add(tab2, text ="Camera Recognition")

        # Expanding Tab content to fit window and include frame
        tabControl.pack(expand = 1, fill ="both", padx= 5, pady=5)

        titleOne = ttk.Label(tab1, text="Photo Recognition", font=("Helvetica",16))
        titleTwo = ttk.Label(tab2, text="Camera Recognition", font=("Helvetica",16))

        titleOne.pack(pady=10)
        titleTwo.pack(pady=10)

        imageFrame = tk.LabelFrame(tab1, text="Display Image")
        imageFrame.pack(padx=5)

        buttonFrame = tk.LabelFrame(tab2, text="Video Stream")
        buttonFrame.pack(padx=5)

        photoLabel = tk.Label(imageFrame, image=self.photo)
        photoLabel.grid(row=0, column=0, columnspan=3, padx=10, pady=10)

        backBtn = tk.Button(imageFrame, state="disabled", text="<<", font=("Helvetica",16), command=lambda: self.backward(photoLabel, imageFrame, backBtn, forwardBtn, textFrame)) #, command=lambda: back(1)
        observeBtn = tk.Button(imageFrame, text="Click to Observe", font=("Helvetica",16), command=lambda: self.observe(cardLabel, textFrame))
        forwardBtn = tk.Button(imageFrame, text=">>", font=("Helvetica",16), command=lambda: self.forward(photoLabel, imageFrame, backBtn, forwardBtn, textFrame)) #, command=lambda: forward(1) # 

        backBtn.grid(row=1, column=0, pady=10)
        observeBtn.grid(row=1, column=1, pady=10)
        forwardBtn.grid(row=1, column=2, pady=10)

        textFrame = tk.LabelFrame(tab1)

        cardLabel = tk.Label(textFrame, font=("Helvetica",16))
        cardLabel.grid(row=1, column=3, sticky = "e", pady=10, ipadx=10)

        cameraBtn = tk.Button(buttonFrame, text="Click here to begin Camera Stream", font=("Helvetica",16), command=lambda: self.ml.startStream())
        cameraBtn.grid(row=1, column=1, pady=240, padx=155)

    def forward(self, photoLabel, imageFrame, backBtn, forwardBtn, textFrame):
        '''
        This method increases the imageNumber index by +1, 
        subsequently updating the image on display (photoLabel) 
        as the next image in my list of images.
        params:
            photoLabel: tkinter gui widget,
        returns: 
            updated photo and image number index
        '''
        
        self.imageNumber += 1
        self.photo = self.ip.displayImage(self.imageNumber)
        
        photoLabel.grid_forget()

        photoLabel = tk.Label(imageFrame, image=self.photo)
        photoLabel.grid(row=0, column=0, columnspan=3, padx=10, pady=10)

        textFrame.pack_forget()

        self.checkState(backBtn, forwardBtn)

        return self.imageNumber

    def backward(self, photoLabel, imageFrame, backBtn, forwardBtn, textFrame):
        '''
        This method decreases the imageNumber index by -1, 
        subsequently updating the image on display (photoLabel) 
        as the previous image in my list of images.
        params:
            photoLabel: tkinter gui widget,
        returns: 
            updated photo and image number index
        '''
        
        self.imageNumber -= 1
        self.photo = self.ip.displayImage(self.imageNumber)
        
        photoLabel.grid_forget()

        photoLabel = tk.Label(imageFrame, image=self.photo)
        photoLabel.grid(row=0, column=0, columnspan=3, padx=10, pady=10)

        textFrame.pack_forget()

        self.checkState(backBtn, forwardBtn)

        return self.imageNumber

    def checkState(self, backBtn, forwardBtn):
        '''
        This method is responsible for ensuring the back button and forward button act accordingly:
        - backBtn is disable when imageNumber index is 0.
        - forwardBtn is disabled when the imageNumber index is at the end of the imageFiles list length.
        '''
        if self.imageNumber == 0:
            #print("disable backBtn")
            backBtn['state'] = tk.DISABLED
        else:
            backBtn['state'] = tk.NORMAL

        if self.imageNumber == self.ip.listLength - 1:
            #print("disable forwardBtn")
            forwardBtn['state'] = tk.DISABLED
        else:
            forwardBtn['state'] = tk.NORMAL

    
    def observe(self, cardLabel, textFrame):

        # Reading the image from the imageFiles list by using its index number (imageNumber)
        img = self.ip.imageFiles[self.imageNumber]
        
        # Getting threshold image (binarisation)
        imgThresh = self.ip.threshold(img)

        # Using morphology to remove the image noise on binarised image
        imgMorph = self.ip.morphology(imgThresh)
        
        # Getting the card (perimeter) contour
        croppingCoordinates, contours = self.ip.edgeDetection(imgMorph)

        # Cropping the image to just the card by using the card contour
        imgCropped = self.ip.cropImage(croppingCoordinates, img)

        # Rotating the cropped image 
        imgRotated = self.ip.rotateImage(imgCropped)

        # Converting the rotated image to grayscale
        imgGray = cv2.cvtColor(imgRotated, cv2.COLOR_BGR2GRAY)
        
        # Get the cards BGR tuple
        colourBar, colourBGR = self.ml.getCardColour(imgRotated)

        # Get Card colour name using KNN classifier
        colourName = self.ml.getColourName(colourBGR)

        # Getting the description list which contains each train image
        desList = self.ip.findDes()

        # Getting the ID of the image (grayscale)
        id = self.ml.findID(imgGray, desList)

        #cv2.imshow('Threshold Image', imgThresh) # To display threshold image
        #cv2.imshow('Morphed Image', imgMorph) # To display morphed image
        #cv2.imshow('Cropped Image', imgCropped) # To display cropped image
        #print(croppingCoordinates) # Print cropping coordinates
        #cv2.imshow('Rotated Image', imgRotated) # To display rotated image

        textFrame.pack_forget()

        cardLabel.config(text="")

        textFrame.pack(padx=5, pady=5)

        if id != -1 and colourName != "N/A":
            card = self.ml.cardNames[id]
            if card in self.ml.textCards:
                if str(card) == "Wild":
                    # Alternative method to retrieve colour for wild (difficult card)
                    colourName = self.ml.getAltCardColour(imgRotated)
                    message = "This is a {} {} Uno card!!".format(colourName, card)
                else:
                    message = "This is a {} {} Uno card!".format(colourName, card)
            else:
                message = "This is a {} number {} Uno card!".format(colourName, card)

        elif colourName == "N/A":
            message = "The colour of this card has not been recognised, please try again."
        else:
            message = "Card not regocnised please try again."    

        cardLabel['text'] = message

        #cv2.waitKey(0)

    def showGui(self):
        self.configureGui()
        self.root.mainloop()


if __name__ == "__main__":
    print("hello")
    root = Gui()
    root.showGui()

        