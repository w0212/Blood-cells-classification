# -*- coding: utf-8 -*-
'''
@Time    : 2020/12/4 10:36
@Author  : Junfei Sun
@Email   : sjf2002@sohu.com
@File    : prepare.py
'''

import cv2
import glob
import os
import numpy as np
import pandas as pd

def _get_data(img_dir):
    dfs=[]
    labels=os.path.join(img_dir,'labels.csv')
    df = pd.read_csv(labels, sep=',')
    df=df.dropna()
    df["Image"] = df['Image'].apply(lambda x: os.path.join(img_dir, '%s.jpeg' %str(x)))
    dfs.append(df)
    train_df = pd.concat(dfs, ignore_index=True)
    n_classes = len(set(train_df['Category']))
    print('Number of training images : {:>5}'.format(train_df.shape[0]))
    print('Number of classes         : {:>5}'.format(n_classes))
    return train_df

def get_feature(data, feature=None, test_split=0.2, seed=113):
    target=[]
    labels=[]
    for c in range(len(data)):
        im = cv2.imread(data['Image'].values[c])
        label=data['Category'].values[c]
        if label!=np.nan:
            labels.append(label)
            target.append(im)

    target = extract_feature(target,feature)
    np.random.seed(seed)
    np.random.shuffle(target)
    np.random.seed(seed)
    np.random.shuffle(labels)
    X_train = target[:int(len(target) * (1 - test_split))]
    y_train = labels[:int(len(target) * (1 - test_split))]

    X_test = target[int(len(target) * (1 - test_split)):]
    y_test = labels[int(len(target) * (1 - test_split)):]


    return (X_train, y_train), (X_test, y_test)


def extract_feature(data, feature):
    """Performs feature extraction
        :param data:data (rows=images, cols=pixels)
        :param feature: which feature to extract
            - None:   no feature is extracted
            - "gray": grayscale features
            - "rgb":  RGB features
            - "hsv":  HSV features
            - "hog":  HOG features
            :returns:       X (rows=samples, cols=features)
        """
    if feature == 'gray':
        target = []
        for t in data:
            if t is not None:
                target.append(cv2.cvtColor(t, cv2.COLOR_BGR2GRAY))
    elif feature == 'hsv':
        target = []
        for t in data:
            if t is not None:
                target.append(cv2.cvtColor(t, cv2.COLOR_BGR2HSV))

    if feature == 'hog':
        imsize=(320,240)
        block_size = (imsize[0] // 4, imsize[1] // 4)
        block_stride = (imsize[0] // 8, imsize[1] // 8)
        cell_size = block_stride
        num_bins = 9
        hog = cv2.HOGDescriptor(imsize, block_size, block_stride, cell_size, num_bins)
        target = [hog.compute(t) for t in target]

    elif feature is not None:
        target=np.array(target).astype(np.float32) / 255
        target = np.array(target).astype(np.float32) / 255

        # subtract mean
        target = [t - np.mean(t) for t in target]

    target=[t.flatten() for t in target]
    return target

def prepare_data(feature_type):
    img_dir = './data/train_data'
    train_data = './feature/train/train_data.npy'
    train_label = './feature/train/train_label.npy'
    val_data = './feature/val/val_data.npy'
    val_label = './feature/val/val_label.npy'

    types=['hog', 'gray', 'hsv', 'rgb']
    if feature_type not in types:
        print('feature type not available')
        exit(0)
    else:
        train_df=_get_data(img_dir)
        (x_train, y_train), (x_valid, y_valid)=get_feature(train_df, feature_type, test_split=0.2, seed=113)
        np.save(train_data, x_train)
        np.save(train_label, y_train)
        np.save(val_data, x_valid)
        np.save(val_label, y_valid)

        data=[]
        data.append(x_train)
        data.append(y_train)
        data.append(x_valid)
        data.append(y_valid)
        return data

