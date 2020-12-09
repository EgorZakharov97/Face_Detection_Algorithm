from FaceDetection import FaceDetector

detector = FaceDetector()

detector.detectFace("./Test/my_face.pgm")

detector.trainModel()

detector.updateSettings(
    detectEdges=True,
    saveData=True,
    showSteps=True,
    mode='no-data',
    singlesDataPath="./singles",
    duplicatesDataPath="./duplicates",
    trainDataPath="./library"
)


