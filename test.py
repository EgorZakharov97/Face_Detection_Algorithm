'''
File that tests the basic functionality of the alrotirhm

'''

from FaceDetection import FaceDetector
import sys
from time import sleep

filename = "./singles/Adrian_Murrell_0001.pgm"
detector = FaceDetector()

print("Edge Detection off")
detector.updateSettings(detectEdges=False, mode='data')
d, closest = detector.detectFace(filename)
d, closest = detector.detectFace(filename)

print("\nTraining for another library")
print("The library should be reconstructed")
detector.updateSettings(trainDataPath="./singles")
detector.trainModel()

print("\nTest the algorithm with several detections")
print("The library should not be reconstructed")
d, closest = detector.detectFace(filename)
d, closest = detector.detectFace(filename)

print("\nTurning on no-data mode, setting back the main library")
detector.updateSettings(mode='no-data', detectEdges=False, trainDataPath='./library')

print("Predicting...\nThe model should be reconstructed")
res = detector.detectFace(filename)
res = detector.detectFace(filename)
