from numpy import *
import matplotlib
import matplotlib.pyplot as plt
import Kmeans
dataSet=Kmeans.loadDataSet("testSet2.txt")
myCentroids,clustAssing = Kmeans.kmeans(dataSet,3)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(list(dataSet[:,0]), list(dataSet[:,1]), marker='^', s=90)
ax.scatter(list(myCentroids[:,0]),list(myCentroids[:,1]),marker='+')
plt.show()