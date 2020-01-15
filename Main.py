import LoadData
import CreateNeuralModel
import FitModel


def main():
    #MnistModel()
    #Cifar10Model()
    #Cifar100Model()
    LetterRecognitionModel()
    #LoadData.LoadLetterRecognitionData()

def MnistModel():
    x_train, x_test, y_train, y_test, input_shape, num_classes = LoadData.LoadMnistData()
    model = CreateNeuralModel.CreateMnistModel(input_shape, num_classes)
    print(model)
    FitModel.FitMnistModel(model, x_train, y_train, x_test, y_test)

def Cifar10Model():
    x_train, x_test, y_train, y_test, num_classes = LoadData.LoadCifar10Data()
    model = CreateNeuralModel.CreateCifar10Model(num_classes, x_train)
    FitModel.FitCifar10Model(model, x_train, y_train, x_test, y_test)

def Cifar100Model():
    x_train, x_test, y_train, y_test, num_classes = LoadData.LoadCifar100Data()
    model = CreateNeuralModel.CreateCifar100Model(num_classes, x_train)
    FitModel.FitCifar100Model(model, x_train, y_train, x_test, y_test)

def LetterRecognitionModel():
    num_classes = 26
    x_train, x_test, y_train, y_test = LoadData.LoadLetterRecognitionData()
    model = CreateNeuralModel.CreateLetterRecignitionModel(num_classes)
    FitModel.FitLetterRecognitionModel(model, x_train.values, y_train, x_test.values, y_test)


if __name__ == '__main__':
    main()