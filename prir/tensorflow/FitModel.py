from __future__ import print_function

import keras
from sklearn.model_selection import KFold
from prir.tensorflow import LoadData, CreateNeuralModel
from timeit import default_timer as timer


def FitMnistModel(crossValidationFlag):
    x_data, y_data, input_shape, num_classes = LoadData.LoadMnistData()

    batch_size = 128
    epochs = 12

    losses = []
    accuracies = []
    if crossValidationFlag:
        n_split = 10
        start = timer()
        for train_index, test_index in KFold(n_split).split(x_data):
            x_train, x_test = x_data[train_index], x_data[test_index]
            y_train, y_test = y_data[train_index], y_data[test_index]
            model = CreateNeuralModel.CreateMnistModel(input_shape, num_classes)

            model.fit(x_train, y_train,
                      batch_size=batch_size,
                      epochs=epochs,
                      verbose=2,
                      validation_data=(x_test, y_test))
            score = model.evaluate(x_test, y_test, verbose=1)
            losses.append(score[0])
            accuracies.append(score[1])
            print('Test loss:', score[0])
            print('Test accuracy:', score[1])
        print('Total Test loss:', sum(losses) / len(losses))
        print('Total Test accuracy:', sum(accuracies) / len(accuracies))
    else:
        start = timer()
        x_train, x_test, y_train, y_test = LoadData.SplitData(x_data, y_data, num_classes)
        model = CreateNeuralModel.CreateMnistModel(input_shape, num_classes)

        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=2,
                  validation_data=(x_test, y_test))
        score = model.evaluate(x_test, y_test, verbose=1)
        losses.append(score[0])
        accuracies.append(score[1])
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
    end = timer()
    print(end - start)


def FitCifar10Model(crossValidationFlag):
    x_data, y_data, num_classes = LoadData.LoadCifar10Data()
    batch_size = 128
    epochs = 12

    losses = []
    accuracies = []

    if crossValidationFlag:
        n_split = 10
        start = timer()
        for train_index, test_index in KFold(n_split).split(x_data):
            x_train, x_test = x_data[train_index], x_data[test_index]
            y_train, y_test = y_data[train_index], y_data[test_index]
            model = CreateNeuralModel.CreateCifar10Model(num_classes, x_data)

            model.fit(x_train, y_train,
                      batch_size=batch_size,
                      epochs=epochs,
                      verbose=2,
                      validation_data=(x_test, y_test))

            scores = model.evaluate(x_test, y_test, verbose=1)
            losses.append(scores[0])
            accuracies.append(scores[1])
            print('Test loss:', scores[0])
            print('Test accuracy:', scores[1])
        print('Total Test loss:', sum(losses) / len(losses))
        print('Total Test accuracy:', sum(accuracies) / len(accuracies))
    else:
        start = timer()
        x_train, x_test, y_train, y_test = LoadData.SplitData(x_data, y_data, num_classes)
        model = CreateNeuralModel.CreateCifar10Model(num_classes, x_data)

        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=2,
                  validation_data=(x_test, y_test))

        scores = model.evaluate(x_test, y_test, verbose=1)
        print('Test loss:', scores[0])
        print('Test accuracy:', scores[1])
    end = timer()
    print(end - start)



def FitCifar100Model(crossValidationFlag):
    x_data, y_data, num_classes = LoadData.LoadCifar100Data()
    batch_size = 128
    epochs = 12
    losses = []
    accuracies = []
    if crossValidationFlag:
        n_split = 10
        start = timer()
        for train_index, test_index in KFold(n_split).split(x_data):
            x_train, x_test = x_data[train_index], x_data[test_index]
            y_train, y_test = y_data[train_index], y_data[test_index]
            model = CreateNeuralModel.CreateCifar100Model(num_classes, x_data)

            model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=2,
                      validation_data=(x_test, y_test))
            score = model.evaluate(x_test, y_test, verbose=1)
            losses.append(score[0])
            accuracies.append(score[1])
            print('Test score:', score[0])
            print('Test accuracy:', score[1])
        print('Total Test loss:', sum(losses) / len(losses))
        print('Total Test accuracy:', sum(accuracies) / len(accuracies))
    else:
        start = timer()
        x_train, x_test, y_train, y_test = LoadData.SplitData(x_data, y_data, num_classes)
        model = CreateNeuralModel.CreateCifar100Model(num_classes, x_data)

        model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=2, validation_data=(x_test, y_test))
        scores = model.evaluate(x_test, y_test, verbose=1)
        print('Test loss:', scores[0])
        print('Test accuracy:', scores[1])
    end = timer()
    print(end - start)


def FitLetterRecognitionModel(crossValidationFlag):
    x_data, y_data, num_classes = LoadData.LoadLetterRecognitionData()
    batch_size = 128
    epochs = 300
    losses = []
    accuracies = []
    if crossValidationFlag:
        n_split = 10
        start = timer()
        for train_index, test_index in KFold(n_split).split(y_data):
            x_train, x_test = x_data[train_index], x_data[test_index]
            y_train, y_test = y_data[train_index], y_data[test_index]

            y_train = keras.utils.to_categorical(y_train, num_classes)
            y_test = keras.utils.to_categorical(y_test, num_classes)
            model = CreateNeuralModel.CreateLetterRecignitionModel(num_classes, x_data.shape[1])

            model.fit(x_train, y_train,
                      batch_size=batch_size,
                      epochs=epochs,
                      verbose=2,
                      validation_data=(x_test, y_test))

            score = model.evaluate(x_test, y_test, verbose=1)
            losses.append(score[0])
            accuracies.append(score[1])
            print('Test loss:', score[0])
            print('Test accuracy:', score[1])
        print('Total Test loss:', sum(losses) / len(losses))
        print('Total Test accuracy:', sum(accuracies) / len(accuracies))
    else:
        start = timer()
        model = CreateNeuralModel.CreateLetterRecignitionModel(num_classes, x_data.shape[1])
        x_train, x_test, y_train, y_test = LoadData.SplitDataForLetterRecognition(x_data, y_data, num_classes)
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=2,
                  validation_data=(x_test, y_test))

        score = model.evaluate(x_test, y_test, verbose=1)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
    end = timer()
    print(end - start)
