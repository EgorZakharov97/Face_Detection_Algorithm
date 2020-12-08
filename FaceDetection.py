import numpy as np
from numpy import linalg as la
import math
import imagefuncs_A4 as imf
import os
import sys
import json

'''
FaceDetector
Is a singleton class with only one public method
detectFace()
'''
class FaceDetector:

    # -----------------
    # Private constants
    # -----------------

    __PATH_TO_FACES = 'Faces/'
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

    __instance = None
    __size = None

    __eigenfaces = np.array([])
    __weights = np.array([])
    __mean = np.array([])
    __file_names = []
    __settings = {}

    # ---------------
    # PRIVATE METHODS
    # ---------------

    # Reads filenames inside the library folder
    # Throws an exception if library contants have changed
    def __checkLibForChanges(self):
        current_file_names = os.listdir(self.__PATH_TO_FACES)
        if not current_file_names in self.__file_names:
            print("Library contains changes. Rebuilding...")
            raise Exception('lib', 'changed')

    # Cleans the data and removed the data folder
    def __cleanData(self):
        if os.path.exists('./' + self.__PATH_TO_DATA):
            print("Cleaning up data...")
            if os.path.exists(self.__PATH_TO_DATA + self.__EIGEN_FACES + ".npy"): os.remove(self.__PATH_TO_DATA + self.__EIGEN_FACES + ".npy")
            if os.path.exists(self.__PATH_TO_DATA + self.__FILE_NAMES + ".npy"): os.remove(self.__PATH_TO_DATA + self.__FILE_NAMES + ".npy")
            if os.path.exists(self.__PATH_TO_DATA + self.__MEAN + ".npy"): os.remove(self.__PATH_TO_DATA + self.__MEAN + ".npy")
            if os.path.exists(self.__PATH_TO_DATA + self.__SETTINGS + ".txt"): os.remove(self.__PATH_TO_DATA + self.__SETTINGS + ".txt")
            if os.path.exists(self.__PATH_TO_DATA + self.__WEIGHTS + ".npy"): os.remove(self.__PATH_TO_DATA + self.__WEIGHTS + ".npy")
            os.rmdir('./' + self.__PATH_TO_DATA)

    # Loads the files required for face analysis
    # located at the data path.
    # Throws an exception if any of the required files were not found
    # Or if new settings conflict with the old ones
    def __getData(self):
        # Reading settings
        filehandler = open(self.__PATH_TO_DATA + self.__SETTINGS + '.txt', 'r')
        settings_read = filehandler.read()
        settings_read = json.loads(settings_read)

        if self.__settings["detectEdges"] != settings_read["detectEdges"]:
            raise Exception("settings", "mismatch")

        self.__eigenfaces = np.load(self.__PATH_TO_DATA + self.__EIGEN_FACES + '.npy')
        self.__weights = np.load(self.__PATH_TO_DATA + self.__WEIGHTS + '.npy')
        self.__mean = np.load(self.__PATH_TO_DATA + self.__MEAN + '.npy')
        self.__file_names = np.load(self.__PATH_TO_DATA + self.__FILE_NAMES + '.npy')

        self.__checkLibForChanges()
        
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
        self.__file_names = os.listdir(self.__PATH_TO_FACES)
        d = []
        print(f"\tDetected {len(self.__file_names)} files")

        for i, filename in enumerate(self.__file_names, start=0):
            img = imf.read_image(self.__PATH_TO_FACES + filename)

            if self.__settings["detectEdges"] :
                print(f"\tDetecting edges: {i} of {len(self.__file_names)} ")
                img = self.__detectEdges(img)

            if self.__size == None or len(d) == 0:
                self.__size = len(img.data[0])
                d = np.zeros((self.__size**2, len(self.__file_names)), dtype=np.int32)

            max_shade, data = img
            reshaped = data.reshape(-1)
            d[:,i] = reshaped

        
        return d

    # Computes the data required for face anamysis
    # Saves the data is required
    def __setData(self):
        print("\tReading faces...")
        d = self.__readFaces()
        n = d.shape[1]
        
        print("\tBuilding library...")

        # Calculate the average column x
        self.__mean = d.mean(axis=1)

        # Subtract x from every column of the d x n matrix
        # as a result we get a transpose of L
        LT = (d.transpose() - self.__mean)

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
        self.__weights = [0] * n

        for i in range(n):
            col_L = L[:,i]
            self.__weights[i] = self.__findWeight(self.__eigenfaces, col_L)

        self.__weights = np.array(self.__weights)

        if self.__settings["saveData"] :
            print("\tdone.\n\tSaving data...")
            if not os.path.exists(self.__PATH_TO_DATA): os.mkdir(self.__PATH_TO_DATA)
            np.save(self.__PATH_TO_DATA + self.__EIGEN_FACES + '.npy', self.__eigenfaces)
            np.save(self.__PATH_TO_DATA + self.__WEIGHTS + '.npy', self.__weights)
            np.save(self.__PATH_TO_DATA + self.__MEAN + '.npy', self.__mean)
            np.save(self.__PATH_TO_DATA + self.__FILE_NAMES + '.npy', self.__file_names)

            # saving current settings
            filehandler = open(self.__PATH_TO_DATA + self.__SETTINGS + '.txt', 'w')
            filehandler.write(json.dumps(self.__settings))
            filehandler.close()
        

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
        max_shade, data = img
        data = data.flatten()

        z = data - self.__mean # subtract mean from image data

        w = self.__findWeight(self.__eigenfaces, z)

        distances = [0] * len(self.__weights) # the distance vector

        for i in range(len(self.__weights)):
            distances[i] = la.norm(self.__weights[i] - w)

        d = np.amin(distances) # the minimal distance to a pic from library
        index = np.where(distances == d)[0][0]

        return d, index

    # Updates settings object
    # Receives settings params
    # If saveData=False, cleans the data and throws an exception
    def __updateSettings(self, detectEdges, saveData):
        
        self.__settings["detectEdges"] = detectEdges
        self.__settings["saveData"] = saveData

        if not self.__settings["saveData"]:
            self.__cleanData()
            raise Exception("settings", "saveData=False")

    # Returns a singleton instance
    @staticmethod 
    def getInstance():
        """ Static access method. """
        if FaceDetector.__instance == None:
            FaceDetector()
        return FaceDetector.__instance

    # Public constructor
    def __init__(self):
        """ Virtually private constructor. """
        if FaceDetector.__instance != None:
            raise Exception("This class is a singleton!")
        else:
           FaceDetector.__instance = self

    # --------------
    # PUBLIC METHODS
    # --------------

    # Runs the face detection algorithm
    def detectFace(self, filename, detectEdges=True, saveData=True):
        print("\n----------------")
        print("Face Or Not Face")
        print("----------------\n")
        print(f"Testing image {filename}\n")

        try:
            self.__updateSettings(detectEdges, saveData)
            print("Loading data...")
            self.__getData()
        except Exception as e:
            # If we catch an exception, we need to rebuild the library
            # We want to rebuild the library if one of the files is missing
            # Or if we received the setting saveData=False
            print("Setting library...")
            self.__setData()
        finally:
            # Test an image
            print("Testing...\n")
            distance, index_in_names_array = self.__testImage(filename)

            print("The closest image is filename=" + self.__file_names[index_in_names_array])
            print(f"The distance is d={distance}\n")
