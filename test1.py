from numpy import *
import matplotlib
import matplotlib.pyplot as plt
import Kmeans
dataSet=Kmeans.loadDataSet("testSet2.txt")
myCentroids,clustAssing = Kmeans.kmeans(dataSet,3)
