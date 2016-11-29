from numpy import *

class Perceptron():

    def __init__(self, inputs, targets):

        # ndim - number of array dimensions
        if ndim(inputs) > 1:
            # shape - tuple of array dimensions
            self.nIn = shape(inputs)[1]
        else:
            self.nIn = 1

        if ndim(targets) > 1:
            self.nOut = shape(targets)[1]
        else:
            self.nOut = 1

        self.nData = shape(inputs)[0]

        # Initialise network
        self.weights = random.rand(self.nIn + 1, self.nOut) * 0.1 - 0.05  # set weights [0, 1) * 0.1 - 0.05

    def pcntrain(self, inputs, targets, eta, nIterations):
        # Add the inputs that match the bias node
        inputs = concatenate((inputs, -ones((self.nData, 1))), axis=1)

        # Training
        change = arange(self.nData)
        for n in range(nIterations):
            self.outputs = self.pcnfwd(inputs)
            self.weights += eta * dot(transpose(inputs), targets - self.outputs)

            print("Iteration: ", n)
            print(self.weights)

        activations = self.pcnfwd(inputs)
        print("Final outputs are:")
        print(activations)

            # Randomise order of inputs
            # random.shuffle(change)
            # inputs = inputs[change, :]
            # targets = targets[change, :]
            # return self.weights

    def pcnfwd(self, inputs):
        outputs = dot(inputs, self.weights)
        # Threshold the outputs
        return where(outputs > 0, 1, 0)


inputs = array([[0, 0], [0, 1], [1, 0], [1, 1]])
targets = array([[0], [1], [1], [1]])
eta = 0.25
nIterations = 10
pcp = Perceptron(inputs, targets)
pcp.pcntrain(inputs, targets, eta, nIterations)
