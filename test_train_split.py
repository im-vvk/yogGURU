import os
import shutil
import numpy as np

cwd = os.getcwd()

TRAIN_DATA_DIR = cwd + "/data/yoga_dataset/images/train/"

TEST_DATA_DIR = cwd + "/data/yoga_dataset/images/test/"

asans = os.listdir(TRAIN_DATA_DIR)
RATIO = 0.20

for asan in asans:
    asandir = TRAIN_DATA_DIR+asan+'/'

    imgnames = os.listdir(asandir)
    idx = np.random.randint(0, len(imgnames), int(len(imgnames)*RATIO))

    destpath = TEST_DATA_DIR+asan+'/'
    try:
        os.mkdir(destpath)
    except:
        ...

    for i in idx:
        try:
            shutil.move(asandir+imgnames[i], destpath)
        except:
            ...
