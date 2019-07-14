from keras.models import Sequential, load_model  # to define the model of neural network
from keras.layers import Conv2D, MaxPooling2D  # to implement the process of convolution or pooling
from keras.layers import Activation, Dropout, Flatten, Dense  # 活性化関数, ドロップアウト処理, データ一次元処理, 全結構造
# from keras.utils import np_utils  # to use data
import keras, sys
import numpy as np
from PIL import Image

classes = ["monkey", "boar", "crow"]
num_classes = len(classes)
image_size = 50


def build_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=[59, 50, 3]))
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

    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])  # loss = 損失関数、正解と推定値との誤差

    # to load file
    model = load_model('./animal_cnn_aug.h5')

    return model


def main():
    image = Image.open(sys.argv[1])
    image = image.convert('RGB')
    image = image.resize((image_size, image_size))
    data = np.asarray(image) / 255
    X = []
    X.append(data)
    X = np.array(X)
    model = build_model()

    result = model.predict([X])[0]
    predicted = result.argmax()
    percentage = int(result[predicted] * 100)
    print("{0}({1} %)".format(classes[predicted], percentage))


if __name__ == "__main__":
    main()