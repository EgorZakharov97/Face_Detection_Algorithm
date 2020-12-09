from FaceDetection import FaceDetector
import ProgressBar
import os

file_names = os.listdir('./duplicates')
detector = FaceDetector()
detector.updateSettings(mode='no-data', showSteps=False)
old_faces = 0
new_faces = 0
not_faces = 0

ProgressBar.initializeProgressBar(len(file_names))

for image_path in file_names:
    ProgressBar.increaseProgressBar()
    result = detector.detectFace("./duplicates/" + image_path)
    if result == 0: old_faces += 1
    elif result == 1: new_faces += 1
    else: not_faces += 1

ProgressBar.completeProgressBar()

print(f"\nOld faces:{old_faces}\nNew faces:{new_faces}\nNot faces:{not_faces}")


# def findAverageD(fileNames, path, detect, save):
#     filelist = []

#     with open(fileNames) as libfile:
#         for line in libfile:
#             name = path + line.strip()
#             filelist.append(name)

#     d = 0

#     for i, image in enumerate(filelist, start=0):
#         print(f"Testing {i+1} of {len(filelist)}")
#         print(f"Testing for {image}")
#         d_next, closest_file = detector.detectFace(image, detectEdges = detect, saveData = save, showSteps=False)
#         print(f"Detected closest: {closest_file}\n")
#         d += d_next

#     return d/len(filelist)
    
    

# #find original dmin and dmax
# print("\nTESTING FOR dmin AND dmax\n")
# print("Step 1 of 4")
# print("Finding average for duplicates. edgeDetection: False, save: True\n")
# originalDMin = findAverageD('duplicates.txt', './duplicates/', detect = False, save = True)

# print("\nStep 2 of 4")
# print("Finding average for singles. edgeDetection: False, save: True\n")
# originalDMax = findAverageD('singles.txt', './singles/', detect = False, save = True)
# originalDifference = originalDMax - originalDMin

# #find modified dmin and dmax
# print("\nStep 3 of 4")
# print("Finding average for duplicates. edgeDetection: True, save: True\n")
# modifiedDMin = findAverageD('duplicates.txt', './duplicates/', detect = True, save = True)

# # print("\nStep 4 of 4")
# print("Finding average for singles. edgeDetection: True, save: True\n")
# modifiedDMax = findAverageD('singles.txt', './singles/', detect = True, save = True)
# modifiedDifference = modifiedDMax - modifiedDMin

# print('The original dmin is: ' + str(originalDMin))
# print('The original dmax is: ' + str(originalDMax))
# print('The original difference is: ' + str(originalDifference))
# print('The modified dmin is: ' + str(modifiedDMin))
# print('The modified dmax is: ' + str(modifiedDMax))
# print('The modified difference is: ' + str(modifiedDifference))
