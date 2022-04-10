import glob
import os
from imageio import imread

def load_cellpose_dataset(train, test):

    def load_folder(folder):

        X = []
        y = []

        for file in sorted(glob.glob(os.path.join(folder, '*img.png'))):
            image = imread(file)[:, :, :2]
            X.append(image)

        for file in sorted(glob.glob(os.path.join(folder, '*masks.png'))):
            mask = imread(file)
            y.append(mask)

        return X, y

    X_train, y_train = load_folder(train)
    X_test, y_test = load_folder(test)

    return X_train, y_train, X_test, y_test
