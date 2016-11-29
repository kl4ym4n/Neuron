import os
from pylab import *
from numpy import *
import main

pima = loadtxt('/home/klayman/Documents/pima-indians-diabetes.data', delimiter=',')
print(shape(pima))

# Plot the first and second values for the two classes
# indices0 = where(pima[:, 8] == 0)
# indices1 = where(pima[:, 8] == 1)
#
# ion()
# plot(pima[indices0, 0], pima[indices0, 1], 'go')
# plot(pima[indices1, 0], pima[indices1, 1], 'rx')
# show()

x1 = pima[:, :8]
x2 = pima[:, 8:9]

print("Output on original data")
pcp = main.Perceptron(pima[:, :8], pima[:, 8:9])
pcp.pcntrain(pima[:, :8], pima[:, 8:9], 0.25, 100)
pcp.confmat(pima[:, :8], pima[:, 8:9])



# Various preprocessing steps
pima[where(pima[:, 0] > 8), 0] = 8

pima[where(pima[:, 7] <= 30), 7] = 1
pima[where((pima[:, 7] > 30) & (pima[:, 7] <= 40)), 7] = 2
pima[where((pima[:, 7] > 40) & (pima[:, 7] <= 50)), 7] = 3
pima[where((pima[:, 7] > 50) & (pima[:, 7] <= 60)), 7] = 4
pima[where(pima[:, 7] > 60)] = 5

pima[:, :8] = pima[:, :8]-pima[:, :8].mean(axis=0)
pima[:, :8] = pima[:, :8]/pima[:, :8].var(axis=0)

trainin = pima[::2, :8]
testin = pima[1::2, :8]
traintgt = pima[::2, 8:9]
testtgt = pima[1::2, 8:9]


pcp1 = main.Perceptron(trainin, traintgt)
pcp1.pcntrain(trainin, traintgt, 0.25, 100)
pcp1.confmat(trainin, traintgt)