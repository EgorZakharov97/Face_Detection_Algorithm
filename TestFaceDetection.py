from FaceDetection import FaceDetector

detector = FaceDetector()


def findAverageD(fileNames, path, detect, save):
    filelist = []

    with open(fileNames) as libfile:
        for line in libfile:
            name = path + line.strip()
            filelist.append(name)

    d = detector.detectFace(filelist[0], detectEdges = detect, saveData = save)

    for image in range( 1 ,len(filelist), +1 ):
        d += detector.detectFace(filelist[image], detectEdges = False, saveData = True)
    
    return d/len(filelist)
    
    

#find original dmin and dmax
originalDMin = findAverageD('duplicates.txt', './duplicates/', detect = False, save = False)
originalDMax = findAverageD('singles.txt', './singles/', detect = False, save = True)
originalDifference = originalDMax - originalDMin

#find modified dmin and dmax
modifiedDMin = findAverageD('duplicates.txt', './duplicates/', detect = True, save = False)
modifiedDMax = findAverageD('singles.txt', './singles/', detect = False, save = True)
modifiedDifference = modifiedDMax - modifiedDMin

print('The original dmin is: ' + str(originalDMin))
print('The original dmax is: ' + str(originalDMax))
print('The original difference is: ' + str(originalDifference))
print('The modified dmin is: ' + str(modifiedDMin))
print('The modified dmax is: ' + str(modifiedDMax))
print('The modified difference is: ' + str(modifiedDifference))




