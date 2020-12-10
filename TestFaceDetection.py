from FaceDetection import FaceDetector
import ProgressBar
import os

file_names = os.listdir('./Test')
detector = FaceDetector()
detector.updateSettings(mode='no-data', showSteps=False)
old_faces = 0
new_faces = 0
not_faces = 0



print("Testing old version")
ProgressBar.initializeProgressBar(len(file_names))
detector.updateSettings(detectEdges=False)

for image_path in file_names:
    ProgressBar.increaseProgressBar()
    result = detector.detectFace("./Test/" + image_path)
    if result == 0: old_faces += 1
    elif result == 1: new_faces += 1
    else: not_faces += 1

ProgressBar.completeProgressBar()

iter_1_old_faces = old_faces
iter_1_new_faces = new_faces
iter_1_not_faces = not_faces

print("Testing newer version")
ProgressBar.initializeProgressBar(len(file_names))

detector.updateSettings(detectEdges=True)

old_faces = 0
new_faces = 0
not_faces = 0

for image_path in file_names:
    ProgressBar.increaseProgressBar()
    result = detector.detectFace("./Test/" + image_path)
    if result == 0: old_faces += 1
    elif result == 1: new_faces += 1
    else: not_faces += 1

ProgressBar.completeProgressBar()

iter_2_old_faces = old_faces
iter_2_new_faces = new_faces
iter_2_not_faces = not_faces


print("\nResults of the old algorithm")
print("From 10 duplicates, 10 singles and 5 not faces we have")
print(f"Old faces:{iter_1_old_faces}\tNew faces:{iter_1_new_faces}\tNot faces:{iter_1_not_faces}")
print(f"Misses: {abs(iter_1_not_faces - 5)}")

print("\nResults of the newer algorithm")
print("From 10 duplicates, 10 singles and 5 not faces we have")
print(f"Old faces:{iter_2_old_faces}\tNew faces:{iter_2_new_faces}\tNot faces:{iter_2_not_faces}")
print(f"Misses: {abs(iter_2_not_faces - 5)}")























# def findAverageD(fileNames, path):
#     filelist = []

#     with open(fileNames) as libfile:
#         for line in libfile:
#             name = path + line.strip()
#             filelist.append(name)

#     d = 0

#     for i, image in enumerate(filelist, start=0):
#         print(f"Testing {i+1} of {len(filelist)}")
#         print(f"Testing for {image}")
#         d_next, closest_file = detector.detectFace(image)
#         print(f"Detected closest: {closest_file}\n")
#         d += d_next

#     return d/len(filelist)
    
    

# #find original dmin and dmax
# print("\nTESTING FOR dmin AND dmax\n")
# print("Step 1 of 4")
# print("Finding average for duplicates. edgeDetection: False, save: True\n")
# detector.updateSettings(detectEdges=False, mode='data')
# originalDMin = findAverageD('duplicates.txt', './duplicates/')

# print("\nStep 2 of 4")
# print("Finding average for singles. edgeDetection: False, save: True\n")
# originalDMax = findAverageD('singles.txt', './singles/')
# originalDifference = originalDMax - originalDMin

# #find modified dmin and dmax
# print("\nStep 3 of 4")
# print("Finding average for duplicates. edgeDetection: True, save: True\n")
# detector.updateSettings(detectEdges=True)
# modifiedDMin = findAverageD('duplicates.txt', './duplicates/')

# # print("\nStep 4 of 4")
# print("Finding average for singles. edgeDetection: True, save: True\n")
# modifiedDMax = findAverageD('singles.txt', './singles/')
# modifiedDifference = modifiedDMax - modifiedDMin

# print('The original dmin is: ' + str(originalDMin))
# print('The original dmax is: ' + str(originalDMax))
# print('The original difference is: ' + str(originalDifference))
# print('The modified dmin is: ' + str(modifiedDMin))
# print('The modified dmax is: ' + str(modifiedDMax))
# print('The modified difference is: ' + str(modifiedDifference))
