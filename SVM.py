
# -*- coding: utf-8 -*-
'''
@Time    : 2020/12/4 10:36
@Author  : Junfei Sun
@Email   : sjf2002@sohu.com
@File    : SVM.py
'''

import os
import numpy as np
from sklearn.svm import LinearSVC
import joblib
from prepare import extract_feature

class SVM():
    def __init__(self):
        self.model_path = './model/SVC.model'
        self.train_data = './feature/train/train_data.npy'
        self.train_label = './feature/train/train_label.npy'
        self.val_data = './feature/val/val_data.npy'
        self.val_label = './feature/val/val_label.npy'
        self.svc = LinearSVC()

    def train(self,data):
        svc = LinearSVC()
        x_train, y_train, x_valid, y_valid = data
        svc.fit(x_train, y_train)
        accuracy = round(svc.score(x_valid, y_valid))
        print('Test accuracy of SVC: ', accuracy)
        joblib.dump(svc, self.model_path) #模型持久化
        return accuracy

    def predict(self,image,feature_type):
        target = []
        im=image
        target.append(im)
        target=extract_feature(target,feature_type)
        if os.path.isfile(self.model_path):
            self.svc = joblib.load(self.model_path)
        else:
            print('No SVM module')
        ID=self.svc.predict(target)
        typeID=ID[0]
        print('Cell prediction class ID: ', typeID)
        return typeID
