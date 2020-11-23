import numpy as np

a = np.array([[1,2,3,4], [4,5,6,5], [7,8,9,6]])
b = a.transpose()

print(np.matmul(a, b))