import LoadData
import CreateNeuralModel
import FitModel


def main():
    # MnistModel()
    Cifar10Model()
    #Cifar100Model()
    # LoadData.LoadLetterRecognitionData()



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



if __name__ == '__main__':
    main()