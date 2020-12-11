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
