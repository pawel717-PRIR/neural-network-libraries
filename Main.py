import LoadData
import CreateNeuralModel
import FitModel as fit

crossValidationFlag = True

def main():
    fit.FitMnistModel(crossValidationFlag)


if __name__ == '__main__':
    main()