# balance_data.py

import numpy as np
import pandas as pd
from collections import Counter
from random import shuffle
import cv2

np_load_old = np.load

# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

all_train_data = [
    np.load('training_data-1.npy'),
    np.load('training_data-2.npy'),
    np.load('training_data-3.npy'),
    np.load('training_data-4.npy'),
    np.load('training_data-5.npy'),
]


lefts = []
rights = []
forwards = []

# shuffle(train_data)
for train_data in all_train_data:
    for data in train_data:
        img = data[0]
        choice = data[1]

        # cv2.imshow('test', img)
        # print(choice)
        # if cv2.waitKey(25) & 0xFF == ord('q'):
        #     cv2.destroyWindow()
        #     break

        if choice == [0, 0, 1, 0, 0, 0, 0, 0, 0]:
            lefts.append([img,choice])
        elif choice == [1, 0, 0, 0, 0, 0, 0, 0, 0]:
            forwards.append([img,choice])
        elif choice == [0, 0, 0, 1, 0, 0, 0, 0, 0]:
            rights.append([img,choice])
        # else:
        #     print('no matches')

forwards = forwards[:len(lefts)][:len(rights)]
lefts = lefts[:len(forwards)]
rights = rights[:len(forwards)]

final_data = forwards + lefts + rights
shuffle(final_data)

np.save('final-training_data.npy', final_data)
