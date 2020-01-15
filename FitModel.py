from __future__ import print_function
import keras
from keras.optimizers import SGD, Adadelta, Adagrad


def FitMnistModel(model, x_train, y_train, x_test, y_test):
    batch_size = 128
    epochs = 12

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.SGD,
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

def FitCifar10Model(model, x_train, y_train, x_test, y_test):
    # initiate RMSprop optimizer
    batch_size = 32
    epochs = 100
    opt = keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-6)

    # Let's train the model using RMSprop
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True)

    scores = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])


def FitCifar100Model(model, x_train, y_train, x_test, y_test):
    # initiate RMSprop optimizer
    batch_size = 128
    epochs = 12
    num_classes = 100
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=epochs, verbose=1, validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, show_accuracy=True, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=epochs, verbose=1, validation_data=(x_test, y_test))
    scores = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])

def FitLetterRecognitionModel(model, x_train, y_train, x_test, y_test):
    batch_size = 128
    epochs = 1000
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    letterRecognition = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])