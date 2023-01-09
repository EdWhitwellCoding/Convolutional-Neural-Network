import numpy
import tensorflow as tf
import time
import matplotlib.pyplot as plt
import cv2

class ConvolutionLayer:
    def __init__(self, NumberOfKernals, KernelSize):
        self.NumberOfKernals = NumberOfKernals
        self.KernelSize = KernelSize
        self.Kernals = numpy.random.randn(NumberOfKernals, KernelSize, KernelSize) / (KernelSize**2) # / (KernelSize**2) is standard to normalize the weights/filter values

    def SaveKernals(self):
        numpy.savetxt("Kernals.txt", self.Kernals.reshape(self.Kernals.shape[0], self.KernelSize**2), delimiter=',')

    def KernalOverlays(self, Input):
        self.Input = Input
        for i in range(Input.shape[1] - self.KernelSize + 1):
            for j in range(Input.shape[2] - self.KernelSize + 1):
                KernalOverlay = Input[:, i:i+self.KernelSize, j:j+self.KernelSize]
                yield KernalOverlay, i, j # use yeild else it will ablosutly kill your memory (alternatively you can use a list and append to it)

    def forward(self, Input):
        Output = numpy.zeros((Input.shape[0], self.NumberOfKernals, Input.shape[1] - self.KernelSize + 1, Input.shape[2] - self.KernelSize + 1))
        Area = self.KernelSize**2
        for Kernal, i, j in self.KernalOverlays(Input):
            Result = numpy.dot(Kernal.reshape(Kernal.shape[0], Area), self.Kernals.reshape(self.Kernals.shape[0], Area).T)
            Output[:, :, i, j] = Result

        #display_image(Output[0, 0])
        return Output

    def backward(self, OutputGradient, LearnRate):

        DerivCostKernal = numpy.zeros(self.Kernals.shape) # creates an empty array with the shape of the kernals and with the number of kernals
        for Kernals, i, j in self.KernalOverlays(self.Input): # calls the function that returns each valid kernal overlay
            Gradient = OutputGradient[:, :, i, j]
            #print("Gradient shape: ", Gradient.shape)
            #print("Kernals shape: ", Kernals.shape)
            Product = numpy.tensordot(Gradient, Kernals, axes=([0], [0]))

            Product = Product / self.Input.shape[0]
            DerivCostKernal += Product

        self.Kernals -= LearnRate * DerivCostKernal # update the kernals
        return DerivCostKernal

class PoolingLayer:
    # this just reduces the size of the image by taking the max value of each kernal
    def __init__(self, KernalSize):
        self.KernalSize = KernalSize

    def forward(self, Input):
        self.Input = Input
        Map = numpy.zeros(Input.shape)
        Output = numpy.zeros((Input.shape[0], Input.shape[1],  Input.shape[2] // self.KernalSize, Input.shape[3] // self.KernalSize))
        Area = self.KernalSize**2
        for i in range(Input.shape[2] // self.KernalSize):
            StartI = i * self.KernalSize
            for j in range(Input.shape[3] // self.KernalSize):
                StartJ = j * self.KernalSize
                Kernal = Input[:, :, StartI : (StartI+self.KernalSize), StartJ:(StartJ+self.KernalSize)]
                
                KernalFlat = Kernal.reshape(Kernal.shape[0], Kernal.shape[1], Area)
                #MaxValues = numpy.amax(Kernal, axis = (2,3))
                MaxIndex = numpy.argmax(KernalFlat, axis = 2)
                MaxValues = KernalFlat[numpy.arange(Kernal.shape[0])[:, None], numpy.arange(Kernal.shape[1]), MaxIndex]
                Output[:, :, i, j] = MaxValues
                Mapped = numpy.zeros(KernalFlat.shape)
                Mapped[numpy.arange(Kernal.shape[0])[:, None], numpy.arange(Kernal.shape[1]), MaxIndex] = 1
                Mapped = Mapped.reshape(Kernal.shape)
                Map[:, :, StartI : (StartI+2), StartJ:(StartJ+2)] = Mapped
        self.Map = Map
        #display_image(Output[0, 0])
        return Output

    def backward(self, OutputGradient): # all completely right!!!!!!!
        Map = self.Map
        shape = Map.shape
        MapFlat = Map.flatten()
        OutputGradientFlat = OutputGradient.flatten()
        MapFlat[MapFlat == 1] = OutputGradientFlat
        MapFlat = MapFlat.reshape(shape)

        return MapFlat

class SoftMax:
    def __init__(self, InputSize, OutputSize):
        self.Weights = numpy.random.randn(OutputSize, InputSize) / InputSize# / InputSize is standard to normalize the weights/filter values
        self.Bias = numpy.zeros(OutputSize)

    def Save(self):
        numpy.savetxt("Weights.txt", self.Weights, delimiter = ",")
        numpy.savetxt("Bias.txt", self.Bias, delimiter = ",")

    def forward(self, Input):
        self.Shape = Input.shape #correct

        Input = numpy.rollaxis(Input, 1, 4) # correct
        FlattenedInput = Input.reshape((self.Shape[0], self.Shape[1] * self.Shape[2] * self.Shape[3])) # correct i think

        FlattenedInput = self.Relu(FlattenedInput) # correct

        self.VectorInput = FlattenedInput # correct
        Weights = self.Weights
        Net = numpy.dot(FlattenedInput, Weights.T) + self.Bias# correct

        Eed = numpy.exp(Net.T)
        Sum = numpy.sum(Eed, axis = 0)

        SoftMaxed = (Eed / Sum).T
        return SoftMaxed

    def Relu(self, Z):
        self.Z = Z
        return numpy.maximum(Z, 0)

    def ReluDeriv(self):
        Z = self.Z
        return Z > 0

    def backward(self, CostGradient, LearnRate):
        Inputs = self.VectorInput

        self.Bias -= (numpy.sum(CostGradient.copy(), axis=1) / (CostGradient.shape[1] / LearnRate))

        #DerivWeights = CostGradient.dot(Inputs) / (Inputs.shape[0] / LearnRate)
        DerivWeights = CostGradient.dot(Inputs) / Inputs.shape[0]
        DerivWeights = DerivWeights * LearnRate

        DerivInputs = self.Weights.T.dot(CostGradient)
        DerivInputs = DerivInputs.T * self.ReluDeriv()

        self.Weights -= DerivWeights

        #DerivInputs = DerivInputs.reshape((self.Shape[0], self.Shape[2], self.Shape[3], self.Shape[1]))

        return DerivInputs

def get_accuracy(Outputs, Y):
    return numpy.sum(Outputs == Y)

def get_predictions(A2):
    return numpy.argmax(A2, 0)

def CNN_forward(Input, ExpectedOutput, Layers):
    Output = Input
    for Layer in Layers:
        Output = Layer.forward(Output)
    Predictions = get_predictions(Output.T)
    Accuracy = get_accuracy(Predictions, ExpectedOutput)
    return Output, Accuracy

def CNN_backward(Gradient, Layers, LearnRate = 0.05):
    CostGradient = Gradient
    for Layer in Layers[::-1]:
        if type(Layer) == SoftMax or type(Layer) == ConvolutionLayer:
            CostGradient = Layer.backward(CostGradient, LearnRate)
        elif type(Layer) == PoolingLayer:
            CostGradient = Layer.backward(CostGradient)

    return CostGradient

def CNN_train(Input, ExpectedOutput, Layers, LearnRate = 0.05):
    Output, Accuracy = CNN_forward(Input, ExpectedOutput, Layers)
    one_hot_Y = EncodeLable(numpy.array([ExpectedOutput]))
    Gradient = Output - one_hot_Y
    Gradient = Gradient.T
    gradient = CNN_backward(Gradient, Layers, LearnRate)
    return Accuracy

def AddNoise(image, var):
      mean = 0
      sigma = var**0.5
      gauss = numpy.random.normal(mean,sigma,(image.shape))
      gauss = gauss.reshape(image.shape)
      noisy = image + gauss
      return noisy

def EncodeLable(Y):
    Desired = numpy.zeros((Y.size, 10))
    Desired[numpy.arange(Y.size), Y] = 1 # Dersired is a 2d array the first dimension is covered by the numpy.arange(Y.size) which does the equivilent to the Indexer in a for loop, 
    #the 2nd d is dictated by that index in Y
    return Desired

def Test(X_test, y_test, Layers, Epoch):
    _, Accuracy = CNN_forward(X_test, y_test, Layers)
    print("Accuracy for Epoch ", Epoch, ": ", Accuracy)


def main():
  # Load training data

  # Define the network
  layers = [
    ConvolutionLayer(16,3), # layer with 8 3x3 filters, output (26,26,16)
    PoolingLayer(2), # pooling layer 2x2, output (13,13,16)
    SoftMax(13*13*16, 10) # softmax layer with 13*13*16 input and 10 output
    ] 

  #TestRealImage(layers)
  #input()
  for epoch in range(7):
    print('Epoch {} ->'.format(epoch+1))

    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X_train = X_train / 255
    X_test = X_test / 255
    X_train_noise = AddNoise(X_train, 0.001)
    X_train = numpy.concatenate((X_train, X_train_noise))
    y_train = numpy.concatenate((y_train, y_train))

    permutation = numpy.random.permutation(len(X_train))
    X_train = X_train[permutation]
    y_train = y_train[permutation]

    batchSize = 32
    QuantityOfSections = 10000
    for Section in range(0, len(X_train), QuantityOfSections):
        X_trainbase = X_train[Section:Section+QuantityOfSections]
        y_trainbase = y_train[Section:Section+QuantityOfSections]
        accuracy = 0
        StartTime = time.time()
        for Index in range(0, len(X_trainbase), batchSize):
            batch_X = X_trainbase[Index:Index+batchSize]
            batch_y = y_trainbase[Index:Index+batchSize]
            accuracy += CNN_train(batch_X, batch_y, layers)
            #print("Batch {} completed in {}".format(Index, time.time() - TimeStart))
        EndTime = time.time()
        print("Section {}. accuracy {}, completed in {}".format(Section, accuracy/QuantityOfSections, EndTime - StartTime))
    print("Testing")
    Test(X_test, y_test, layers, epoch)
  input("Press Enter to continue...")
  layers[0].SaveKernals()
  layers[2].Save()

main()