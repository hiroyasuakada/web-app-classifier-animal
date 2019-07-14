from PIL import Image
# glob: to get the list of array 配列の一覧を取得
import os, glob
import numpy as np
# from sklearn import cross_validation
from sklearn import model_selection  # トレーニング＋テストに分割

classes = ["monkey", "boar", "crow"]
num_classes = len(classes)
image_size = 50
num_testdata = 100

# to read image 画像の読み込み
X_train = []  # initialization
X_test = []
Y_train = []
Y_test = []

for index, classlabel in enumerate(classes):
    photos_dir = "./" + classlabel
    files = glob.glob(photos_dir + "/*.jpg")
    for i, file in enumerate(files):
        if i >= 200: break
        image = Image.open(file)
        image = image.convert("RGB")
        image = image.resize((image_size, image_size))
        data = np.asarray(image)  # array 配列に変換

        if i < num_testdata:
            X_test.append(data)
            Y_test.append(index)
        else:
            for angle in range(-20, 20, 5):
                # to rotate
                img_r = image.rotate(angle)
                data = np.asarray(img_r)
                X_train.append(data)
                Y_train.append(index)

                # to reverse
                img_trans = img_r.transpose(Image.FLIP_LEFT_RIGHT)
                data = np.asarray(img_trans)
                X_train.append(data)
                Y_train.append(index)


# X = np.array(X)  # python list → data type TensorFlowが扱いやすいデータ型に揃える
# Y = np.array(Y)

X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(Y_train)
y_test = np.array(Y_test)

# X_train, X_test, y_train, y_test = model_selection.train_test_split(X, Y)
xy = (X_train, X_test, y_train, y_test) # to merge 4 variable into 1
np.save("./animal_aug.npy", xy)  # save array of np as text file

# activate tf140 コマンドプロンプトから呼び出し