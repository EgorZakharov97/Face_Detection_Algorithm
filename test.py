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

# progress_step = 0
# progress_counter = 0
# size = 12

# def initializeProgressBar(size):
#     global progress_step
#     progress_step = 100/size
#     sys.stdout.write("\n")
#     for i in range(100):
#         sys.stdout.write("-")
#     sys.stdout.write("\n")
#     sys.stdout.flush()

# def increaseProgressBar():
#     global progress_counter
#     global progress_step
#     progress_counter += progress_step
#     if progress_counter >= 1:
#         new_progress_counter = int(progress_counter)
#         for i in range(new_progress_counter):
#             sys.stdout.write('|')
#         sys.stdout.flush()
#         progress_counter -= new_progress_counter

# initializeProgressBar(size)

# for i in range(size):
#     increaseProgressBar()
#     sleep(0.01)
