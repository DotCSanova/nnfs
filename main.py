import nnfs
from neuralLayers import DenseLayer
from activationFunctions import Activation_ReLU, Activation_Softmax
from lossFunctions import Loss_CategoricalCrossentropy
from nnfs.datasets import spiral_data

def main():
    
    nnfs.init()

    X, y = spiral_data(100, 3)

    dense1 = DenseLayer(2, 3)

    activation1 = Activation_ReLU()   

    dense2 = DenseLayer(3, 3)

    activation2 = Activation_Softmax()

    loss_function = Loss_CategoricalCrossentropy()

    dense1.forward(X)

    activation1.forward(dense1.output)

    dense2.forward(activation1.output)

    activation2.forward(dense2.output)

    print(activation2.output[:5])

    loss = loss_function.calculate(activation2.output, y)

    print("Loss:", loss)
    


if __name__ == "__main__":
    main()
