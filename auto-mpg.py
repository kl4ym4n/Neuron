from pylab import *
from numpy import *
import main


inputs = array([[0, 0], [0, 1], [1, 0], [1, 1]])
targets = array([[0], [1], [1], [1]])

pcp = main.Perceptron(inputs, targets)

auto = loadtxt('/home/klayman/Documents/auto-mpg.data', comments='"')

# Separate the data into training and testing sets

# Normalise the data

# This is the training part

trainin = auto[::2, :7]
testin = auto[1::2, :7]
traintgt = auto[::2, 7:8]
testtgt = auto[1::2, 7:8]

beta = pcp.linreg(trainin, traintgt)
testin = concatenate((testin, -ones((shape(testin)[0], 1))), axis=1)
testout = dot(testin, beta)
error = sum((testout - testtgt)**2)
print(error)