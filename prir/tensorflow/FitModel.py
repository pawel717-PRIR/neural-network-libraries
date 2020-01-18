from __future__ import print_function
from sklearn.model_selection import KFold
from prir.tensorflow import LoadData, CreateNeuralModel
from timeit import default_timer as timer

import tensorflow as keras


def FitMnistModel(crossValidationFlag):
    x_data, y_data, input_shape, num_classes = LoadData.LoadMnistData()
    model = CreateNeuralModel.CreateMnistModel(input_shape, num_classes)
    batch_size = 256
    epochs = 12

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    if crossValidationFlag:
        n_split = 10
        start = timer()
        for train_index, test_index in KFold(n_split).split(x_data):
            x_train, x_test = x_data[train_index], x_data[test_index]
            y_train, y_test = y_data[train_index], y_data[test_index]

            model.fit(x_train, y_train,
                      batch_size=batch_size,
                      epochs=epochs,
                      verbose=2,
                      validation_data=(x_test, y_test))
            score = model.evaluate(x_test, y_test, verbose=1)
            print('Test loss:', score[0])
            print('Test accuracy:', score[1])

    else:
        start = timer()
        x_train, x_test, y_train, y_test = LoadData.SplitData(x_data, y_data, num_classes)
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


def FitCifar10Model(crossValidationFlag):
    x_data, y_data, num_classes = LoadData.LoadCifar10Data()
    model = CreateNeuralModel.CreateCifar10Model(num_classes, x_data)
    batch_size = 256
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
                      verbose=2,
                      validation_data=(x_test, y_test))

            scores = model.evaluate(x_test, y_test, verbose=1)
            print('Test loss:', scores[0])
            print('Test accuracy:', scores[1])
    else:
        start = timer()
        x_train, x_test, y_train, y_test = LoadData.SplitData(x_data, y_data, num_classes)
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
    model = CreateNeuralModel.CreateCifar100Model(num_classes, x_data)
    batch_size = 256
    epochs = 12

    if crossValidationFlag:
        n_split = 10
        start = timer()
        for train_index, test_index in KFold(n_split).split(x_data):
            x_train, x_test = x_data[train_index], x_data[test_index]
            y_train, y_test = y_data[train_index], y_data[test_index]

            model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=epochs, verbose=2,
                      validation_data=(x_test, y_test))
            score = model.evaluate(x_test, y_test, show_accuracy=True, verbose=1)
            print('Test score:', score[0])
            print('Test accuracy:', score[1])
    else:
        start = timer()
        x_train, x_test, y_train, y_test = LoadData.SplitData(x_data, y_data, num_classes)
        model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=epochs, verbose=2, validation_data=(x_test, y_test))
        scores = model.evaluate(x_test, y_test, verbose=1)
        print('Test loss:', scores[0])
        print('Test accuracy:', scores[1])
    end = timer()
    print(end - start)


def FitLetterRecognitionModel(crossValidationFlag):
    x_data, y_data, num_classes = LoadData.LoadLetterRecognitionData()
    model = CreateNeuralModel.CreateLetterRecignitionModel(num_classes, x_data.shape[1])
    batch_size = 256
    epochs = 200

    if crossValidationFlag:
        n_split = 10
        start = timer()
        for train_index, test_index in KFold(n_split).split(y_data):
            x_train, x_test = x_data[train_index], x_data[test_index]
            y_train, y_test = y_data[train_index], y_data[test_index]

            y_train = keras.utils.to_categorical(y_train, num_classes)
            y_test = keras.utils.to_categorical(y_test, num_classes)

            model.fit(x_train, y_train,
                      batch_size=batch_size,
                      epochs=epochs,
                      verbose=2,
                      validation_data=(x_test, y_test))

            score = model.evaluate(x_test, y_test, verbose=1)
            print('Test loss:', score[0])
            print('Test accuracy:', score[1])
    else:
        start = timer()
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
