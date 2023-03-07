import numpy as np
import datetime
from numba import njit
from .encoder import Encoder

@njit
def runInputPerceptron(weights, input):
    sum = np.sum(weights[input])
    return 0 if sum <= 0 else sum

class Neural:
    def __init__(self, settings={}):
        self.settings = settings
        self.settings['maxIterations'] = self.settings.get('maxIterations', 150)
        self.settings['learningRate'] = self.settings.get('learningRate', 0.002)
        self.log = self.settings.get('log', False)

    def prepareCorpus(self, corpus):
        self.encoder = self.settings.get('encoder', Encoder(self.settings.get("processor")))
        self.encoded = self.encoder.encodeCorpus(corpus)

    def initialize(self, corpus):
        self.prepareCorpus(corpus)
        self.perceptrons = [
            {
                'intent': intent,
                'id': self.encoder.intentMap.get(intent),
                'weights': np.zeros(self.encoder.numFeature, dtype=np.float32)
            }
            for intent in self.encoder.intents
        ]

    def trainPerceptron(self, perceptron, data, learningRate):
        weights = perceptron['weights']
        error = 0
        for d in data:
            input, output = d[0], d[1]
            actualOutput = runInputPerceptron(weights, input)
            expectedOutput = 1 if output == perceptron['id'] else 0
            currentError = expectedOutput - actualOutput
            if currentError:
                error += currentError ** 2
                change = currentError * learningRate
                for key in input:
                    weights[key] += change
        return error

    def train(self, corpus):
        hrstarttotal = datetime.datetime.now()
        self.initialize(corpus)
        data = self.encoded["train"]
        maxIterations = self.settings['maxIterations']
        learningRate = self.settings['learningRate']
        iterations = 0
        while iterations < maxIterations:
            iterations += 1
            hrstart = datetime.datetime.now()
            error = 0
            for perceptron in self.perceptrons:
                error += self.trainPerceptron(perceptron, data, learningRate)
            error = error / (len(data) * len(self.perceptrons))
            if self.log:
              hrend = datetime.datetime.now()
              elapsed = (hrend - hrstart).total_seconds() * 1000
              print(f"Epoch {iterations} loss {error} time {elapsed}ms")
        hrendtotal = datetime.datetime.now()
        elapsed = (hrendtotal - hrstarttotal).total_seconds() * 1000
        print(f"Epoch {iterations} loss {error} time {elapsed}ms")
