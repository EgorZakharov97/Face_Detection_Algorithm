from FaceDetection import testImage


def findAverageD(fileNames, path):
    filelist = []

    with open(fileNames) as libfile:
        for line in libfile:
            name = path + line.strip()
            filelist.append(name)

    dmin = 0

    for image in range( len(filelist) ):
        d, index = testImage(filelist[image])
        dmin += d

    return dmin/len(filelist)
    



#find modified dmin and dmax
modifiedDMin = findAverageD('duplicates.txt', './duplicates/')
modifiedDMax = findAverageD('singles.txt', './singles/')

modifiedDifference = modifiedDMax - modifiedDMin
print('The modified difference is: ' + str(modifiedDifference))
