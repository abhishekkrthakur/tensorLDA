import matplotlib.pyplot as plt
import numpy as np

predict_vec = np.load('hist_data.dat')
true_labels = np.load('labels_orig.dat')

plt.figure()

negIdx = np.where(true_labels ==0)[0]
posIdx = np.where(true_labels ==1)[0]
print negIdx.shape
print posIdx.shape
negTrain = predict_vec[negIdx]
posTrain = predict_vec[posIdx]
plt.hist(negTrain,histtype='stepfilled',label = 'negative')
plt.hist(posTrain,histtype='stepfilled',label= 'positive')
plt.legend()
#plt.savefig("hist.png")
plt.show()

