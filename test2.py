import Kmeans
from numpy import *
import matplotlib.pyplot as plt
dataSet=Kmeans.loadDataSet("testSet.txt")
centList,myNewAssments = Kmeans.biKmeans2(dataSet,4)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(list(dataSet[:,0]), list(dataSet[:,1]), marker='^',c= (myNewAssments[:,0]*210).tolist(), s=90,alpha = 0.5)
ax.scatter(list(centList[:,0]),list(centList[:,1]),marker='+')
plt.show()

