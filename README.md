# Uno Card Recognition Project 

By Patrick J Nkwocha

The aim of this project is to compose code used for recognising cards from the game UNO. This project is written using the programming language python and utilises openCV for computer vision, and sklearn for machine learning.

The libraries used in this project are:

-   numpy: for data structures
-   os: for file management
-   cv2: for computer vision functions
-   tkinter: for the GUI (Graphical User Interface)
-   sklearn: for classification using kmeans clustering and k neareest neighbours
-   joblib: for storing and retrieving data
-   PIL: for displaying images

Without these libraries installed the project will not succesfully run.

The python files which are used in this project are:

-   main.py
-   image_processing.py
-   machine_learning.py

I decided to take an OOP approach for the construction of my code to make it easier to identify object-event relationships.

This project is capable of the following:

- Recognising card color
- Differentiating the card from background
- Identify card number
- Creating and modifying datasets

# main.py

THIS IS THE FILE TO RUN!

To run this project it is important to only run the main.py file. This can be done via the terminal by writing 'python main.py' or by opening the 'main.py' in your chosen IDE and clicking the run button .

The class Gui inside this file (main.py) creates a GUI and executes the code as the user interacts with the inteface.

The GUI has been split into two tabs:

-   Photo Recognition
-   Camera Recognition

The self explainatory aspects of the system can be accessed by clicking on either tab.

- # Photo Recognition 

    This tab is responsible for displaying the feedback given from the system when querying image files.

It has 3 buttons:

    - forwardBtn ">>" - assisting the user in moving forward through the images directory. 
    - backwardsBtn "<<" - assisting the user in moving backward through the images directory.
    - observeBtn "Click to Observe" - This button is responsible for triggering events (functions) which well subsequently demonstrate the system in action (CLICK THIS BUTTON TO SEE THE PHOTO RECOGNITION ASPECT OF THE SYSTEM IN ACTION).

- # Camera Recognition

This tab is responsible for displaying the feedback given from the system when querying from a camera.

This tab has 1 button:
    - cameraBtn "Click here to start camera stream" - This button is responsible for triggering the event which begins the camera stream. At this point the user is then instructed on how to intiate the observation of a card (pressing the enter key) or close the stream (pressing the space key).

# Potential Improvements
I believe I could improve this system by making use of neural networks in opposition to KNN Clustering. This is something I began to attempt which the beginnings can be found in the #breakdown folder under the name "cnn.py". I could also use more training images for my ORB description list. This would provide a wider scope for feature recogniton. The camera system can be improved as its colour recognition seems inconsistent. I believe I have changed something without taking notice, as results are very different from previous results.

# image_processing.py

This file is mainly responsible for handling the vision elements of the system. Within this file openCV image objects undergo various proccesses via the ImageProccessing class to make it easier to be pericieved by the machine learning aspect.

Initially, as an object is created the images for training and testing are loaded as openCV image objects and placed into seperate lists for later use. Simultaneously an event within the configuration of the gui is triggered to display the firs entry in the images dataset. The majority of other functions are all triggered in response to interaction with the interface.

# machine_learning.py

This file is responsible for recieving the proccessed images/caamera stream and using KMeans Clustering and KNeighborsClassifier to cluster existing training and testing datasets and classifying our image based on preprocessing findings

- # Other File descriptions:

    # Practical_learning folder
    - reading.py -  Reading Images, videos, and webcam
    - basic_functions.py -  Basic functions for OpenCV
    - resize_cropping.py - Resizing and Cropping
    - colour_detection.py - Alternative working solution for card colour detection
    - feature_recognition.py - Utilizing ORB algorithm to determin keypoint matches amongst 2 images
    - cnn.py - Beginings of neural network implementation for feature/text recognition. (To run this code move it into the main directory)
    - vision_elements.py - Breakdown of vision elements utilized in image processing
    - classifier.py - Beginnings of classifier, which would be used in machine learning aspect for colour matching.

    # images folder -
        images provided by lecturer Dr Eris for openCV dataset

    # trainImages folder-
        images used for training dataset

    # trainingData folder-
        images used for training dataset for cnn.py (neural networks implementation attempt) THIS WAS NOT USED for this project.

    # c_classifier.sav 
        joblib pickle file used to save and load classifer dataset.

Thank you!

# Sources

https://www.stackvidhya.com/save-and-load-machine-learning-models-to-disk/#:~:text=Loading%20The%20Model%20Using%20JobLib%20To%20classify%20the,load%20the%20saved%20model%20using%20joblib.load%20%28%29%20method.
https://joblib.readthedocs.io/en/latest/auto_examples/parallel_memmap.html#sphx-glr-auto-examples-parallel-memmap-py
https://www.tutorialspoint.com/using-opencv-with-tkinter
https://www.timpoulsen.com/tag/opencv.html
https://www.tutorialspoint.com/keras/keras_installation.htm
https://www.geeksforgeeks.org/python-programming-language/?ref=shm
https://www.geeksforgeeks.org/k-means-clustering-introduction/
https://www.youtube.com/watch?v=nnH55-zD38I
https://www.youtube.com/watch?v=WQeoO7MI0Bs
