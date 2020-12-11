import numpy as np
from numpy import linalg as la
import ProgressBar
import math
import imagefuncs_A4 as imf
import os
import sys
import json

'''
FACE DETECTOR CLASS

'''
class FaceDetector:

    # -----------------
    # Private constants
    # -----------------
    __PATH_TO_DATA = 'data/'
    __EIGEN_FACES = 'eigenfaces'
    __WEIGHTS = 'weights'
    __MEAN = 'mean'
    __FILE_NAMES = 'file_names'
    __SETTINGS = 'settings'

    __SIGMA = 2
    __R = 4
    __GAUSS2 = imf.gaussian1D(__SIGMA, __R)

    # -----------------
    # Private variables
    # -----------------

    _instance = None
    _size = None

    _path_to_train_data = './library'

    _eigenfaces = np.array([])
    _weights = np.array([])
    _mean = np.array([])
    _file_names = []            # Holds an array of filenames in training dataset
    _settings = {               # Holds current application settings
        "detectEdges":      True,                   # Should the algorithm use Canney Edge Detection
        "saveData":         True,                   # Should the algorithm save data into a file for the future use
        "showSteps":        True,                   # Would you like to see print statements
        "important_setting_update": False,          # Is true when change in serrings requires the library to be rebuilt
        "mode":             "data",                 # "data" will return distance and closest file name; "no-data" will return 0 is image is in the library, -1 if not a face and 1 if a new face
        "singlesDataPath":     "./singles",         # path to library with single faces
        "duplicatesDataPath":   "./duplicates",     # path to library with face duplicates
        "trainDataPath":    "./library"             # path to training library
    }
    _data_set = False                   # Tracks if up to date data was loaded from the file
    _settings_set = False               # Tracks if up to date settings were loaded from the file

    # ---------------
    # PRIVATE METHODS
    # ---------------

    # Reads filenames inside the library folder
    # Throws an exception if library contants have changed
    def __checkLibForChanges(self):
        current_file_names = os.listdir(self._settings["trainDataPath"])
        if not current_file_names in self._file_names:
            if self._settings["showSteps"]: print("Library contains changes. Rebuilding...")
            raise Exception('lib', 'changed')

    # Writes computed model into a file
    def __saveData(self):
        if self._settings["showSteps"]: print("done.\nSaving data...")
        if not os.path.exists(self.__PATH_TO_DATA): os.mkdir(self.__PATH_TO_DATA)
        np.save(self.__PATH_TO_DATA + self.__EIGEN_FACES + '.npy', self._eigenfaces)
        np.save(self.__PATH_TO_DATA + self.__WEIGHTS + '.npy', self._weights)
        np.save(self.__PATH_TO_DATA + self.__MEAN + '.npy', self._mean)
        np.save(self.__PATH_TO_DATA + self.__FILE_NAMES + '.npy', self._file_names)
        self.__saveSettings()
    
    # Removes computed model and removes the folder
    def __cleanData(self):
        if os.path.exists('./' + self.__PATH_TO_DATA):
            if self._settings["showSteps"]: print("Cleaning up data...")
            if os.path.exists(self.__PATH_TO_DATA + self.__EIGEN_FACES + ".npy"): os.remove(self.__PATH_TO_DATA + self.__EIGEN_FACES + ".npy")
            if os.path.exists(self.__PATH_TO_DATA + self.__FILE_NAMES + ".npy"): os.remove(self.__PATH_TO_DATA + self.__FILE_NAMES + ".npy")
            if os.path.exists(self.__PATH_TO_DATA + self.__MEAN + ".npy"): os.remove(self.__PATH_TO_DATA + self.__MEAN + ".npy")
            if os.path.exists(self.__PATH_TO_DATA + self.__SETTINGS + ".txt"): os.remove(self.__PATH_TO_DATA + self.__SETTINGS + ".txt")
            if os.path.exists(self.__PATH_TO_DATA + self.__WEIGHTS + ".npy"): os.remove(self.__PATH_TO_DATA + self.__WEIGHTS + ".npy")
            os.rmdir('./' + self.__PATH_TO_DATA)
            self._data_set = False

    # Writes settings into a file
    def __saveSettings(self):
        if not os.path.exists(self.__PATH_TO_DATA): os.mkdir(self.__PATH_TO_DATA)
        filehandler = open(self.__PATH_TO_DATA + self.__SETTINGS + '.txt', 'w')
        filehandler.write(json.dumps(self._settings))
        filehandler.close()

    # Loads the files required for face analysis
    # located at the data path.
    # Throws an exception if any of the required files were not found
    # Or if new settings conflict with the old ones
    def __loadData(self):
        try:
            self._file_names = np.load(self.__PATH_TO_DATA + self.__FILE_NAMES + '.npy')
            if self._settings["important_setting_update"]: raise Exception('lib', 'modified')
            self.__checkLibForChanges()
            if self._data_set: return
        except:
            self._data_set = False
            raise Exception('lib', 'needs erconstruction')

        if not self._settings_set:
            # Reading settings
            filehandler = open(self.__PATH_TO_DATA + self.__SETTINGS + '.txt', 'r')
            settings_read = filehandler.read()
            self._settings = json.loads(settings_read)

        self._eigenfaces = np.load(self.__PATH_TO_DATA + self.__EIGEN_FACES + '.npy')
        self._weights = np.load(self.__PATH_TO_DATA + self.__WEIGHTS + '.npy')
        self._mean = np.load(self.__PATH_TO_DATA + self.__MEAN + '.npy')

        self._data_set = True
        
    # Returns a new image that contains the edges
    # Of an imput image
    def __detectEdges(self, image):
        smooth_image = imf.convolve2D_separable(image, self.__GAUSS2)
        gradientjk = imf.gradient_vectors(smooth_image.data)
        grad_lengths = imf.gradient_lengths(gradientjk)
        grad_max = np.amax(grad_lengths)
        grad_image = imf.PGMFile(grad_max, grad_lengths)
        thin_data = imf.thin_edges(gradientjk, grad_lengths)
        thin_image = imf.PGMFile(grad_max, thin_data)
        low_threshold = 0.1*thin_image.max_shade
        high_threshold = 0.18*thin_image.max_shade
        suppr_data = imf.suppress_noise(thin_image.data, low_threshold, high_threshold)
        suppr_image = imf.PGMFile(thin_image.max_shade, suppr_data)
        return suppr_image

    # Reads faces in the library folder and detects its edges if required
    # Returns a matrix containing the data of all images
    def __readFaces(self):
        train_data_path = self._settings["trainDataPath"]
        self._file_names = os.listdir(train_data_path)
        d = [] # contains all the images data

        if self._settings["showSteps"]: 
            print(f"Detected {len(self._file_names)} files, processing")
            ProgressBar.initializeProgressBar(len(self._file_names))

        for i, filename in enumerate(self._file_names, start=0):
            img = imf.read_image(train_data_path + '/' + filename)

            # Detect edges if settings say to do so
            if self._settings["detectEdges"] :
                img = self.__detectEdges(img)

            # If image size was not initialized, write the first image size
            # !!! Assumption: all images in the folder have the same size !!!
            if self._size == None or len(d) == 0:
                self._size = len(img.data[0])
                d = np.zeros((self._size**2, len(self._file_names)), dtype=np.int32)

            max_shade, data = img
            reshaped = data.reshape(-1)
            d[:,i] = reshaped

            if self._settings["showSteps"]: ProgressBar.increaseProgressBar()

        if self._settings["showSteps"]: ProgressBar.completeProgressBar()

        
        return d    

    # finds eigenvalues, eigenvectors, and corresponding eigenfaces
    # returns a matrix containing eigenfaces
    def __findEigenFaces(self, covMatrix, L):

        #find eigenvalues and eigenvectors and sort them in ascending order
        eigenValues, eigenVectors = la.eig(covMatrix)

        idx = eigenValues.argsort()[::-1]   
        eigenValues = eigenValues[idx]
        eigenVectors = eigenVectors[:,idx]

        # #find k eigenvalues such that they sum to 9k% of the total
        eigenValSum = 0.95 * np.sum(eigenValues )
        sum = 0
        k = 0
        while( sum < eigenValSum and eigenValues[k]+sum < eigenValSum):
            sum += eigenValues[k]
            k = k + 1

        #edge case for when the sum is less than the largest eigenvalue 
        if(k == 0):
            eigenValues = eigenValues[0:1]
            eigenVectors = eigenVectors[:, 0:1]

        else:
            eigenValues = eigenValues[0:k]
            eigenVectors = eigenVectors[:, 0:k]

        #find eigenfaces
        eigenFaces = [] 

        for colNum in range(0, len(eigenVectors[0])):
            temp = eigenVectors[:, colNum:colNum+1]
            temp = np.matmul(L, temp)
            temp = temp / la.norm(temp)
            eigenFaces.append( temp )

        return np.array(eigenFaces)

    # returns the weight vector for column Lj of matrix j
    def __findWeight(self, eigenFaces, Lj):
        weightVector = [0] * len(eigenFaces)
        xj = Lj.flatten()

        for i in range(0, len(eigenFaces)):
            #get column slice from eigenFaces and flatten
            vi = eigenFaces[i].flatten()
        
            weightVector[i] = np.dot( xj, vi )

        return np.array(weightVector)

    # Uses previously calculated data to test
    # Whether an image is a face
    def __testImage(self, file_name_to_test):

        img = imf.read_image(file_name_to_test)
        if self._settings["detectEdges"]: img = self.__detectEdges(img)

        max_shade, data = img
        data = data.flatten()

        z = data - self._mean # subtract mean from image data

        w = self.__findWeight(self._eigenfaces, z)

        distances = [0] * len(self._weights) # the distance vector

        for i in range(len(self._weights)):
            distances[i] = la.norm(self._weights[i] - w)

        d = np.amin(distances) # the minimal distance to a pic from library
        index = np.where(distances == d)[0][0]

        return d, index

    # Computes d_low and d_high
    def __findBoundaries(self):
        if self._settings["showSteps"]: print("Looking for boundaries...")
        test_data_path = self._settings["singlesDataPath"]
        file_names = os.listdir(test_data_path)
        d = 0

        if self._settings["showSteps"]:
            print("Calculating d_high...")
            ProgressBar.initializeProgressBar(len(file_names))

        for i, image_path in enumerate(file_names, start=0):
            if self._settings["showSteps"]: ProgressBar.increaseProgressBar()
            d_new, index = self.__testImage(test_data_path + "/" + image_path)
            d += d_new

        if self._settings["showSteps"]: ProgressBar.completeProgressBar()

        d_high = d/len(file_names)

        test_data_path = self._settings["duplicatesDataPath"]
        file_names = os.listdir(test_data_path)
        d = 0

        if self._settings["showSteps"]:
            print("Calculating d_low...")
            ProgressBar.initializeProgressBar(len(file_names))

        for i, image_path in enumerate(file_names, start=0):
            if self._settings["showSteps"]: ProgressBar.increaseProgressBar()
            d_new, index = self.__testImage(test_data_path + "/" + image_path)
            d += d_new

        if self._settings["showSteps"]: ProgressBar.completeProgressBar()

        d_low = d/len(file_names)

        self._settings["d_low"] = d_low
        self._settings["d_high"] = d_high

        if self._settings["saveData"]: self.__saveSettings()

    # Returns the final answer
    def __makeConclusion(self, d):
        try:
            d_low = self._settings["d_low"]
            d_high = self._settings["d_high"]
        except:
            self.__findBoundaries()
            d_low = self._settings["d_low"]
            d_high = self._settings["d_high"]

        if(d <= d_low):
            # The test image is an old face
            if self._settings["showSteps"]: print("Old face")
            return 0
        elif(d_low < d <= d_high):
            # The test image is a new face
            if self._settings["showSteps"]: print("New face")
            return 1
        else:
            # The test image is not a face
            if self._settings["showSteps"]: print("Not a face")
            return -1

    # Returns a singleton instance
    @staticmethod 
    def getInstance():
        """ Static access method. """
        if FaceDetector._instance == None:
            FaceDetector()
        return FaceDetector._instance

    # Public constructor
    def __init__(self):
        """ Virtually private constructor. """
        if FaceDetector._instance != None:
            raise Exception("This class is a singleton!")
        else:
            try:
                self.__loadData()
            except:
                pass
            FaceDetector._instance = self

    # --------------
    # PUBLIC METHODS
    # --------------

    # Computes the data required for face anamysis
    # Saves the data is required
    def trainModel(self):
        if self._settings["showSteps"]: print("TRAINING THE MODEL\nReading faces...")
        d = self.__readFaces()
        n = d.shape[1]
        
        if self._settings["showSteps"]: print("Building library...")

        # Calculate the average column x
        self._mean = d.mean(axis=1)

        # Subtract x from every column of the d x n matrix
        # as a result we get a transpose of L
        LT = (d.transpose() - self._mean)

        # find L
        L = LT.transpose()

        # find LTL by matrix multiplication
        LTL = np.matmul(LT, L)

        # divide LTL by (n-1)
        multiplier = 1/(n-1)
        LTL = multiplier * LTL

        # find eigenfaces
        self._eigenfaces = self.__findEigenFaces(LTL, L)

        # find weights
        self._weights = [0] * n

        for i in range(n):
            col_L = L[:,i]
            self._weights[i] = self.__findWeight(self._eigenfaces, col_L)

        self._weights = np.array(self._weights)

        if self._settings["saveData"] : self.__saveData()

        # Turn off the flag for rebuilding model
        self._settings["important_setting_update"] = False
        if self._settings["saveData"]: self.__saveSettings()

        if self._settings["mode"] == 'no-data': self.__findBoundaries()

        

    # Updates settings object
    # Receives settings params
    # Settings:
    #           detectEdges - whether images requires edges detection
    #           saveData    - whether you want to save the processed library for future use
    #           showSteps   - whether you want to see print statements
    def updateSettings(self, detectEdges=None, saveData=None, showSteps=None, singlesDataPath=None, duplicatesDataPath=None, trainDataPath=None, mode=None):
        
        if detectEdges != None:
            if detectEdges != self._settings["detectEdges"]: self._settings["important_setting_update"] = True
            self._settings["detectEdges"] = detectEdges

        if saveData != None:           self._settings["saveData"] = saveData
        if showSteps != None:          self._settings["showSteps"] = showSteps
        if singlesDataPath != None:    self._settings["singlesDataPath"] = singlesDataPath
        if duplicatesDataPath != None:    self._settings["duplicatesDataPath"] = duplicatesDataPath
        if trainDataPath != None:      self._settings["trainDataPath"] = trainDataPath

        if mode == 'data': self._settings["mode"] = "data"
        elif mode == 'no-data': self._settings["mode"] = "no-data"

        if(self._settings["saveData"]): self.__saveSettings()

    # Runs the face detection algorithm
    def detectFace(self, filename):
        if self._settings["showSteps"]: print(f"Testing image {filename}\n")

        try:
            if not self._settings["saveData"]:
                self.__cleanData()
                raise Exception("settings", "saveData=False")

            self.__loadData()
        except Exception as e:
            # If we catch an exception, we need to rebuild the library
            # We want to rebuild the library if one of the files is missing
            # Or if we received the setting saveData=False
            self.trainModel()
        finally:
            # Test an image

            if self._settings["showSteps"]: print("Testing...\n")
            d, index_in_names_array = self.__testImage(filename)

            if self._settings["showSteps"]: print("done.")

            if self._settings["mode"] == 'no-data':
                return self.__makeConclusion(d)
            elif self._settings["mode"] == 'data':
                if self._settings["showSteps"]: print("The closest image is filename=" + self._file_names[index_in_names_array])
                if self._settings["showSteps"]: print(f"The distance is d={d}\n")
                return d, self._file_names[index_in_names_array]
