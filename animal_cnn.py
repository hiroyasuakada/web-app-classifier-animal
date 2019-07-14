from keras.models import Sequential  # to define the model of neural network
from keras.layers import Conv2D, MaxPooling2D  # to implement the process of convolution or pooling
from keras.layers import Activation, Dropout, Flatten, Dense  # 活性化関数, ドロップアウト処理, データ一次元処理, 全結構造
from keras.utils import np_utils  # to use data
import keras
import numpy as np

classes = ["monkey", "boar", "crow"]
num_classes = len(classes)
image_size = 50

# to define main function


def main():
    X_train, X_test, y_train, y_test = np.load("./animal.npy", allow_pickle=True)  # to load data
    X_train = X_train.astype("float") / 256  # 整数→浮動小数点数でnormalize
    X_test = X_test.astype("float") / 256
    y_train = np_utils.to_categorical(y_train, num_classes)  # one-hot-vector 正解１、他は０
    y_test = np_utils.to_categorical(y_test, num_classes)

    model = model_train(X_train, y_train)
    model_eval(model, X_test, y_test)


def model_train(X, y):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=X.shape[1:]))
    model.add(Activation('relu'))  # 活性化関数
    model.add(Conv2D(32, (3, 3)))  # second layer
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))  # third layer
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())  # to arrange data in a line
    model.add(Dense(512))  # the output of fully connected layer
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3))
    model.add(Activation('softmax'))  # ニューラルネットワークの出力結果を確立に変換

    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)  # トレーニング時の更新algorithm、最適化の手法

    model.compile(loss='categorical_crossentropy',
                    optimizer=opt, metrics=['accuracy'])  # loss = 損失関数、正解と推定値との誤差

    model.fit(X, y, batch_size=32, epochs=100)

# to save file
    model.save('./animal_cnn.h5')

    return model


def model_eval(model, X, y):
    scores = model.evaluate(X, y, verbose=1)
    print('Test Loss:', scores[0])
    print('Test Accuracy:', scores[1])


if __name__ == "__main__":
    main()





