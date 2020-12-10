from FaceDetection import FaceDetector

detector = FaceDetector()

detector.updateSettings(
    detectEdges=False,
    saveData=True,
    showSteps=True,
    mode='no-data',
    singlesDataPath="./singles",
    duplicatesDataPath="./duplicates",
    trainDataPath="./library"
)

detector.detectFace("./Test/Robert_De_Niro_0002.pgm")