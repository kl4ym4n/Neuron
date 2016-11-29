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

            # print("Iteration: ", n)
            # print(self.weights)
            #
            # activations = self.pcnfwd(inputs)
            # print("Final outputs are:")
            # print(activations)

            # Randomise order of inputs
            # random.shuffle(change)
            # inputs = inputs[change, :]
            # targets = targets[change, :]
            # return self.weights

    def pcnfwd(self, inputs):
        outputs = dot(inputs, self.weights)
        # Threshold the outputs
        return where(outputs > 0, 1, 0)

    def confmat(self, inputs, targets):

        # Add the inputs that match the bias node
        inputs = concatenate((inputs, -ones((self.nData, 1))), axis=1)
        outputs = dot(inputs, self.weights)

        nClasses = shape(targets)[1]

        if nClasses == 1:
            nClasses = 2
            outputs = where(outputs > 0, 1, 0)
        else:
            # 1-of-N encoding
            outputs = argmax(outputs, 1)
            targets = argmax(targets, 1)

        cm = zeros((nClasses, nClasses))
        for i in range(nClasses):
            for j in range(nClasses):
                cm[i, j] = sum(where(outputs == i, 1, 0) * where(targets == j, 1, 0))

        print(cm)
        print(trace(cm) / sum(cm))


inputs = array([[0, 0], [0, 1], [1, 0], [1, 1]])
inputs2 = array([[0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 0]])
targets = array([[0], [1], [1], [1]])
targets2 = array([[0], [1], [1], [0]])
eta = 0.25
nIterations = 15
pcp = Perceptron(inputs, targets)
pcp.pcntrain(inputs, targets, eta, nIterations)
