import numpy as np
import math
import imagefuncs_A3 as imf
import os

SIZE = 200
MAX_SHADE = 255
BORDER = 20
PATH = 'Faces/'

def readFaces():
    file_names = os.listdir(PATH)
    d = np.zeros((SIZE**2, len(file_names)), dtype=np.int32)

    for i, filename in enumerate(file_names, start=0):
        img = imf.read_image(PATH + filename)
        max_shade, data = img
        reshaped = data.reshape(-1)
        d[:,1] = reshaped

    return d
    

def step1():
    print("Reading faces...")
    d = readFaces()
    # d = np.array([[1,2,3],[4,5,6],[7,8,9]])
    
    print("Pricessing...")
    # Calculate the average column x
    mean = d.mean(axis=1)
    # Subtract x from every column of the d x n matrix
    L = (d.transpose() - mean).transpose()
    


step1()