from __future__ import print_function
import keras
from sklearn.model_selection import KFold
import CreateNeuralModel
import LoadData
from timeit import default_timer as timer


def FitMnistModel(crossValidationFlag):
    x_data, y_data, input_shape, num_classes = LoadData.LoadMnistData()
    model = CreateNeuralModel.CreateMnistModel(input_shape, num_classes)
    batch_size = 128
    epochs = 12

    if crossValidationFlag:
        n_split = 10
        start = timer()
        for train_index, test_index in KFold(n_split).split(x_data):
            x_train, x_test = x_data[train_index], x_data[test_index]
            y_train, y_test = y_data[train_index], y_data[test_index]

            model.fit(x_train, y_train,
                      batch_size=batch_size,
                      epochs=epochs,
                      verbose=1,
                      validation_data=(x_test, y_test))

            score = model.evaluate(x_test, y_test, verbose=0)
            print('Test loss:', score[0])
            print('Test accuracy:', score[1])

    else:
        start = timer()
        x_train, x_test, y_train, y_test = LoadData.SplitData(x_data, y_data, num_classes)
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  validation_data=(x_test, y_test))

        score = model.evaluate(x_test, y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
    end = timer()
    print(end - start)

def FitCifar10Model(crossValidationFlag):
    x_data, y_data, input_shape, num_classes = LoadData.LoadCifar10Data()
    model = CreateNeuralModel.CreateCifar10Model(num_classes, x_data)
    batch_size = 32
    epochs = 100

    if crossValidationFlag:
        n_split = 10
        start = timer()
        for train_index, test_index in KFold(n_split).split(x_data):
            x_train, x_test = x_data[train_index], x_data[test_index]
            y_train, y_test = y_data[train_index], y_data[test_index]

            model.fit(x_train, y_train,
                      batch_size=batch_size,
                      epochs=epochs,
                      validation_data=(x_test, y_test),
                      shuffle=True)

            scores = model.evaluate(x_test, y_test, verbose=1)
            print('Test loss:', scores[0])
            print('Test accuracy:', scores[1])
    else:
        start = timer()
        x_train, x_test, y_train, y_test = LoadData.SplitData(x_data, y_data, num_classes)
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(x_test, y_test),
                  shuffle=True)

        scores = model.evaluate(x_test, y_test, verbose=1)
        print('Test loss:', scores[0])
        print('Test accuracy:', scores[1])
    end = timer()
    print(end - start)



def FitCifar100Model(crossValidationFlag):
    x_data, y_data, input_shape, num_classes = LoadData.LoadCifar100Data()
    model = CreateNeuralModel.CreateCifar10Model(num_classes, x_data)
    batch_size = 128
    epochs = 12

    if crossValidationFlag:
        n_split = 10
        start = timer()
        for train_index, test_index in KFold(n_split).split(x_data):
            x_train, x_test = x_data[train_index], x_data[test_index]
            y_train, y_test = y_data[train_index], y_data[test_index]

            model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=epochs, verbose=1,
                      validation_data=(x_test, y_test))
            score = model.evaluate(x_test, y_test, show_accuracy=True, verbose=0)
            print('Test score:', score[0])
            print('Test accuracy:', score[1])
    else:
        start = timer()
        x_train, x_test, y_train, y_test = LoadData.SplitData(x_data, y_data, num_classes)
        model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=epochs, verbose=1, validation_data=(x_test, y_test))
        scores = model.evaluate(x_test, y_test, verbose=1)
        print('Test loss:', scores[0])
        print('Test accuracy:', scores[1])
    end = timer()
    print(end - start)


def FitLetterRecognitionModel(crossValidationFlag):
    x_data, y_data, input_shape, num_classes = LoadData.LoadLetterRecognitionData()
    model = CreateNeuralModel.CreateLetterRecignitionModel(num_classes)
    batch_size = 128
    epochs = 1000

    if crossValidationFlag:
        n_split = 10
        start = timer()
        for train_index, test_index in KFold(n_split).split(x_data):
            x_train, x_test = x_data[train_index], x_data[test_index]
            y_train, y_test = y_data[train_index], y_data[test_index]

            model.fit(x_train, y_train,
                      batch_size=batch_size,
                      epochs=epochs,
                      verbose=1,
                      validation_data=(x_test, y_test))

            score = model.evaluate(x_test, y_test, verbose=0)
            print('Test loss:', score[0])
            print('Test accuracy:', score[1])
    else:
        start = timer()
        x_train, x_test, y_train, y_test = LoadData.SplitDataForLetterRecognition(x_data, y_data, num_classes)
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  validation_data=(x_test, y_test))

        score = model.evaluate(x_test, y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
    end = timer()
    print(end - start)