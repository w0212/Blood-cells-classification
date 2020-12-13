# -*- coding: utf-8 -*-
'''
@Time    : 2020/12/4 10:36
@Author  : Junfei Sun
@Email   : sjf2002@sohu.com
@File    : main.py
'''

import cv2
from SVM import SVM
from prepare import prepare_data

cell_class = {11:'EOSINOPHIL',
              22:'LYMPHOCYTE',
              33:'MONOCYTE',
              44:'NEUTROPHIL'}

types = ['hog', 'gray', 'rgb', 'hsv']
feature_type = types[3]

img_path = './data/test_data/LYMPHOCYTE/_0_1050.jpeg'  #adjust the test image
img=cv2.imread(img_path)
cv2.putText(img,'LYMPHOCYTE',(23,45),cv2.FONT_HERSHEY_COMPLEX,0.6,(0,0,255),1)
cv2.imshow('Result',img)
cv2.waitKey()

svm=SVM()
data=prepare_data(feature_type)
print('data:',data)

svm.train(data)
img=cv2.imread(img_path)
ID_num=svm.predict(img,feature_type)

