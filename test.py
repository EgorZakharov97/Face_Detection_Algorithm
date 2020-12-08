from FaceDetection import FaceDetector
import sys

file_to_test = "./singles/Adrian_Murrell_0001.pgm"

if len(sys.argv) > 1:
    file_to_test = sys.argv[1]

detector = FaceDetector()

detector.detectFace(file_to_test, detectEdges=True, saveData=True)
