import tensorLDA as tl
import numpy as np
import matplotlib.pyplot as plt
from sklearn import (metrics, cross_validation, linear_model, preprocessing)
from sklearn.metrics import metrics
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
import pylab as pl

traindata = np.load('train_data_new.dat')
labels = np.load('labels_new.dat')
y = np.load('labels_orig.dat')
X = traindata
W = tl.getW(traindata, labels, 0.0001, 9)
plt.imshow(W,cmap='Greys_r')
plt.show()
tl.slidingWindow('/Users/abhishek/Documents/workspace/tensor_LDA/Test/TEST_25.PGM',W, 3.0)



#tl.XM(W,traindata,labels,3)
#tl.testOnImage(W,'/Users/abhishek/Documents/workspace/tensor_LDA/Test/TEST_0.PGM')
#tl.slidingWindow('/Users/abhishek/Documents/workspace/tensor_LDA/Test/TEST_41.PGM',W)

#===============================================================================
# # Calculate precision and recall
# X_train, X_cv, y_train, y_cv = cross_validation.train_test_split(X, y, test_size=.40)
# W = tl.getW(traindata, labels, 0.0001, 9)
# pred = tl.predict(X_cv,W,4.0)
# 
# accuracy = metrics.accuracy_score(y_cv,pred)
# precision = metrics.precision_score(y_cv,pred)
# recall = metrics.recall_score(y_cv,pred)
# 
# precision, recall, thresholds = precision_recall_curve(y_cv, pred)
# area = auc(recall, precision)
# print("Area Under Curve: %0.2f" % area)
# 
# pl.clf()
# pl.plot(recall, precision, label='Precision-Recall curve')
# pl.xlabel('Recall')
# pl.ylabel('Precision')
# pl.ylim([0.0, 1.05])
# pl.xlim([0.0, 1.0])
# pl.title('AUC=%0.2f' % area)
# pl.legend(loc="lower left")
# pl.show()
# 
# print accuracy, precision, recall
# 
#         
#===============================================================================