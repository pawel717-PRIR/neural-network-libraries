from prir.tensorflow import FitModel as fit

# Uncomment line below to run on cpu
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

crossValidationFlag = False


def main():
    print("fit.FitCifar10Model(True)")
    print("fit.FitMnistModel(True)")
    fit.FitMnistModel(crossValidationFlag)
    fit.FitCifar10Model(crossValidationFlag)
    print("fit.FitCifar10Model(True)")
    fit.FitCifar10Model(crossValidationFlag)
    print("fit.FitLetterRecognitionModel(True)")
    fit.FitLetterRecognitionModel(crossValidationFlag)


if __name__ == '__main__':
    main()
