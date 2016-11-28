class Perceptron():
    def fire(self, nData, N, M, activation, weights, inputs):
        for data in range(nData):
            for n in range(N):
                activation[data][0] = 0
                for m in range(M+1):
                    activation[data][n] = weights[m][n] * inputs[data][m]

                if activation[data][n] > 0:
                    activation[data][n] = 1
                else:
                    activation[data][n] = 0

        return activation
