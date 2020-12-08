from FaceDetection import FaceDetector

detector = FaceDetector()


def findAverageD(fileNames, path, detect, save):
    filelist = []

    with open(fileNames) as libfile:
        for line in libfile:
            name = path + line.strip()
            filelist.append(name)

    d = 0
    hit_counter = 0

    for i, image in enumerate(filelist, start=0):
        print(f"Testing {i} of {len(filelist)}")
        print(f"Testing for {image}")
        d_next, closest_file = detector.detectFace(image, detectEdges = False, saveData = True, showSteps=False)
        print(f"Detected closest: {closest_file}\n")
        d += d_next
        if closest_file in image: hit_counter += 1

    print(f"Hit rate: {hit_counter/len(filelist)}")
    return d/len(filelist)
    
    

#find original dmin and dmax
print("\nTESTING FOR dmin AND dmax\n")
print("Step 1 of 4")
print("Finding average for duplicates. edgeDetection: False, save: False\n")
originalDMin = findAverageD('duplicates.txt', './duplicates/', detect = False, save = False)

print("\nStep 2 of 4")
print("Finding average for singles. edgeDetection: False, save: True\n")
originalDMax = findAverageD('singles.txt', './singles/', detect = False, save = True)
originalDifference = originalDMax - originalDMin

#find modified dmin and dmax
print("\nStep 3 of 4")
print("Finding average for duplicates. edgeDetection: True, save: False\n")
modifiedDMin = findAverageD('duplicates.txt', './duplicates/', detect = True, save = False)

print("\nStep 4 of 4")
print("Finding average for singles. edgeDetection: False, save: True\n")
modifiedDMax = findAverageD('singles.txt', './singles/', detect = False, save = True)
modifiedDifference = modifiedDMax - modifiedDMin

print('The original dmin is: ' + str(originalDMin))
print('The original dmax is: ' + str(originalDMax))
print('The original difference is: ' + str(originalDifference))
print('The modified dmin is: ' + str(modifiedDMin))
print('The modified dmax is: ' + str(modifiedDMax))
print('The modified difference is: ' + str(modifiedDifference))




