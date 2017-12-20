import random
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle

np.random.seed(104)

def sigmoid(x):
    return 1/(1 + math.exp(-x))
sigmoid.deriv = lambda x: x * (1-x)

def relu(x):
    return max(x,0)
relu.deriv = lambda x: 1 if x > 0 else 0

class Network:
    def __init__(self, layers, inputs):
        self.activation = relu
        self.batchSize = 100
        self.batches = 10
        self.learningRate = 0.01
        self.lessons = 0
        self.layers = layers
        self.weights=[]
        self.learning=[]
        Prev_inputs = inputs
        for layer in layers:
            self.weights.append(np.random.rand(Prev_inputs + 1, layer)-0.5)
            self.learning.append(np.zeros((Prev_inputs + 1, layer)))
            Prev_inputs = layer

    def getDerivs(self, activations, sums, outputs):
        derivs = [2*np.subtract(activations[-1], outputs)]
        # print('activations:===================')
        # for layer, act in enumerate(activations):
        #     print(act)
        #     print(str(layer) + '========')
        # print('sums:===================')
        # for layer, act in enumerate(sums):
        #     print(act)
        #     print(str(layer) + '========')
        # print('weights:===================')
        # for index, weight in enumerate(self.weights):
        #     print(weight)
        #     print(str(index) + '========')
        weightDerivs = []
        for l in reversed(range(0, len(self.weights))):
            derivBias = []
            for i in range(0,len(activations[l+1])):
                derivBias.append(self.activation.deriv(sums[l+1][i])* derivs[0][i])
            derivBias = [derivBias]

            derivCols = []
            for i in range(0,len(activations[l+1])):
                derivCol = []
                for j in range(0, len(activations[l])):
                    derivCol.append(
                        activations[l][j] * 
                        self.activation.deriv(sums[l+1][i]) * 
                        derivs[0][i]
                    )
                derivCols.append(derivCol)

            derivCols = np.array(derivCols).T
            curDerivs = np.concatenate((derivBias, derivCols), 0)
            weightDerivs.insert(0,curDerivs)
            derivCol = []
            for i in range(0, len(activations[l])):
                dz = 0
                for j in range(0, len(derivs[0])):
                    dz += self.weights[l][i+1][j]*derivs[0][j]
                derivCol.append(dz)
            derivs.insert(0,derivCol)
        return weightDerivs

    def activate(self, inputs):
        activations = inputs
        for layer in self.weights:
            activations = list(map(
                    lambda w: self.activation(np.dot(w.T, [1] + activations)[0]),
                    np.split(layer,len(layer[0]),1)
                    ))
        return activations

    def loss(self, inputs, output):
        return np.linalg.norm(np.subtract(self.activate(inputs), output))/2

    def losses(self, pairs):
        total = 0
        for pair in pairs:
            total += self.loss(pair[0], pair[1])
        return total/len(pairs)

    def trainFunc(self, function):
        for i in range(0,self.batches):
            for i in range(0,self.batchSize):
                inp = np.random.random() + 0.5
                net.train([inp], [math.sin(inp)+1])

    def train(self, inputs, outputs):
        activations = [inputs]
        sums = [inputs]
        for layer in self.weights:
            sums.append(list(map(
                    lambda w: np.dot(w.T, [1] + activations[-1])[0],
                    np.split(layer,len(layer[0]),1)
                    )))
            activations.append(list(map(
                    lambda w: self.activation(np.dot(w.T, [1] + activations[-1])[0]),
                    np.split(layer,len(layer[0]),1)
                    )))

        weightDerivs = self.getDerivs(activations, sums, outputs)
        cost = sum(np.subtract(activations[-1], outputs)**2)

        for i in range(0, len(weightDerivs)):
            self.learning[i] = self.learning[i] + weightDerivs[i]
        self.lessons += 1
        """
        for i in weightDerivs:
            print( i)
        for i in self.learning:
            print( i)
        """

        return cost

    def learn(self):
        self.learning = map(
                lambda weights: np.multiply(weights , 
                    1.0/ self.lessons
                    * -self.learningRate), 
                self.learning
                )
        self.weights = np.add(self.weights, self.learning)
        self.learning = []
        self.lessons = 0
        Prev_inputs = 1
        for layer in self.layers:
            self.learning.append(np.zeros((Prev_inputs + 1, layer)))
            Prev_inputs = layer

def testFunc(f, name):
    net = Network([20,20,1],1)
    cost = 10000000
    func = f
    errors = list()

    fig = plt.figure()
    while cost > 1:
        cost = 0
        for i in np.arange(-2,2,0.01):
            cost += net.train([i],[func(i)])
        net.learn()
        values = map(lambda x: net.activate([x])[0], np.arange(-2,2,0.01))
        realvalues = map(lambda x: func(x), np.arange(-2,2,0.01))

        plt.plot(np.arange(-2,2,0.01), values, 'r', label='area')
        plt.plot(np.arange(-2,2,0.01), realvalues, 'b', label='area')

        plt.savefig(name + '.png');
        plt.clf()
        errors.append(cost)
        print( cost)

    plt.plot(range(0,len(errors)), errors, 'r', label='area')
    plt.savefig(name + '_progress' + '.png');

if (__name__ == "__main__"):
    # testFunc(lambda x: x + 4, 'line')
    # testFunc(lambda x: x*x, 'square.png')
    testFunc(lambda x: math.sin(2*x) + 1, 'sin')
    # testFunc(lambda x: abs(x), 'vee')
    print( 'done')
