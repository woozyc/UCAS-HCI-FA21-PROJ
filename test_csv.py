import csv
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
array = genfromtxt("D:\\Code\\Python\\hum-com-inter\\world_coor.csv", delimiter=',')
print(np.shape(array))
theta = array[1:, 0]
x = array[1:, 1]
y = array[1:, 2]
z = array[1:, 3]
#print(np.shape(x))

#f = csv.reader(open('D:\\Code\\Python\\hum-com-inter\\world_coor.csv', 'r'))
#for line in f:
#    print(line)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
print(np.shape([1,2,3]))
print(np.shape([[1],[2],[3]]))
plt.show()