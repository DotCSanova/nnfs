import numpy as np

# ReLU activation function
class Activation_ReLU:

    def forward(self, inputs):

        self.inputs = inputs
        self.output = np.maximum(0, inputs)
    
    def backward(self,dvalues):

        dinputs = dvalues.copy()
        self.dinputs[self.inputs <=0] = 0

# Softmax activation function
class Activation_Softmax:

    def forward(self, inputs):

        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilities


        

