# Implementation of Tensor LDA for object detection in images
# Author: Abhishek Thakur
# Based on paper by C. Bauckhage and J.K. Tsotsos


from scipy.linalg import sqrtm, inv
import numpy as np
import random
from PIL import Image
import glob
import os
import matplotlib.pyplot as plt
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
import matplotlib.cm as cm
from skimage.feature import match_template
from sklearn import preprocessing
from numpy import *
from numpy.random import randint,random_integers
from numpy.core.umath_tests import inner1d
from scipy.optimize import minimize, rosen, rosen_der
import scipy
from scipy.signal import convolve2d, correlate2d
from PIL import Image, ImageDraw
from scipy.ndimage.filters import gaussian_filter
from matplotlib.patches import Rectangle
from numpy.lib.stride_tricks import as_strided as ast



def Gram_Schmidt(vecs, row_wise_storage=True, tol=1E-10):
    
    vecs = asarray(vecs)  # transform to array if list of vectors
    if row_wise_storage:
        A = transpose(vecs).copy()
    else:
        A = vecs.copy()

    m, n = A.shape
    V = zeros((m,n))

    for j in xrange(n):
        v0 = A[:,j]
        v = v0.copy()
        for i in xrange(j):
            vi = V[:,i]

            if (abs(vi) > tol).any():
                v -= (vdot(v0,vi)/vdot(vi,vi))*vi
        V[:,j] = v


    return transpose(V) if row_wise_storage else V


def get_trainingdata():
    path = '/Users/abhishek/Documents/workspace/tensor_LDA/Train/'
    datapath = '/Users/abhishek/Documents/workspace/tensor_LDA/Train/*.pgm'
    n = len(glob.glob(datapath))
    traindata = np.empty((n,2511))
    labels = np.empty(n)
    
    tot_count = 0
    count_p = 0
    count_n = 0
    
    for infile in glob.glob( os.path.join(path, '*.pgm') ):
        lbl_str = infile[57:58]
        #print lbl_str
        img = Image.open(infile)
        img = np.asarray(img)
        img = np.hstack(img)
        traindata[tot_count] = img
        if (lbl_str=='P'):
            labels[tot_count] = 1
            count_p += 1
        else:
            labels[tot_count] = -1
            count_n += 1
        print tot_count
        tot_count += 1
        
        
    for i in range(n):
        if (labels[i] == 1):
            labels[i] /= count_p
        else:
            labels[i] /= count_n
    
    traindata.dump('train_data_new.dat')
    labels.dump('labels_new.dat')  
        
def get_testdata():
    path = '/Users/abhishek/Documents/workspace/tensor_LDA/Test/'
    datapath = '/Users/abhishek/Documents/workspace/tensor_LDA/Test/*.PGM'
    n = len(glob.glob(datapath))
    traindata = np.empty((n,24150))

    tot_count = 0
    
    for infile in glob.glob( os.path.join(path, '*.PGM') ):
        img = Image.open(infile)
        img = np.asarray(img)
        img = np.hstack(img)
        traindata[tot_count] = img
        print tot_count
        tot_count += 1
    traindata.dump('test_data.dat')
     
        
    
def createLabels(trainDir,n):
    labels = np.empty((n,1))
    
    for i in range(n):
        print trainDir   
        

def calc_contraction_u(traindata, u):
    n = len(traindata)
    contraction_data = np.empty((n,81))
    
    for i in range(n):
        img = np.reshape(traindata[i], (31,81))
        #print img.shape, u.shape
        #plt.imshow(img)
        #plt.show()
        #print img
        contraction_data[i] = np.dot(u,img)
        #print contraction_data[i].shape
    #print contraction_data.shape
    return contraction_data
        
def calc_contraction_v(traindata, v):
    n = len(traindata)
    contraction_data = np.empty((n,31))
    for i in range(n):
        img = np.reshape(traindata[i], (31,81))
        #print img.shape, u.shape
        #print i
        temp = np.dot(img,v.T)
        contraction_data[i] = temp.T 
    
    return contraction_data

def dist(x,y):   
    return np.sqrt(np.sum((x-y)**2))

def normalize(arr):
    arr=arr.astype('float32')
    if arr.max() > 1.0:
        arr/=255.0
    return arr        


def qr_mgs( A ):
    """QR decomposition of A.
    
    Modified Gram-Schmidt algorithm, row version (Bjorck alg. 2.1)"""

    A = np.array(A, dtype=float)
    m,n = A.shape
    Q = np.zeros( (m,n) )
    R = np.zeros( (n,n) )

    for k in range( 0, n ) :
        R[k,k] = np.linalg.norm( A[:,k] )
        Q[:,k] = A[:,k] / R[k,k]

        for j in range( k+1, n ) :
            R[k,j] = np.dot( Q[:,k], A[:,j] )
            A[:,j] = A[:,j] - Q[:,k] * R[k,j]
    
    return Q,R



def rho_1(traindata, labels, eps,k):
    #u_init = random_integers(-10,10,size=(1,31)) # change range to R
    u_init =np.random.rand(1,31)
    #v_init = random_integers(0,256,size=(1,81)) # change range to R
    st_u = u_init
    t = 0
    dist2 = 0
    while True:
        t = t + 1
        
        cont_u = calc_contraction_u(traindata,u_init)
        v1 = inv(np.dot(cont_u.T,cont_u))
        v2 = np.dot(cont_u.T,labels)
        v_temp = np.dot(v1,v2)
        #v_temp = normalize(u_temp)
        vp = v_temp
        
        cont_v = calc_contraction_v(traindata,v_temp)
        u1 = inv(np.dot(cont_v.T,cont_v))
        u2 = np.dot(cont_v.T,labels)
        u_temp = np.dot(u1,u2)
        dist1 = np.linalg.norm(u_temp - u_init)
        up = u_temp
        #t_st_u = np.vstack((st_u,u_temp))
        #u_gs = Gram_Schmidt(t_st_u)
            #print st_u.shape
        #u_temp =u_gs[-1]
        u_temp = normalize(u_temp)
            #st_u = np.vstack((st_u,u))
        up = u_temp
        if(abs(dist2 - dist1) < eps):
            break
        u_init = u_temp
        print t, abs(dist2-dist1)
        dist2 = dist1 
    #print u_temp.shape, v_temp.shape
    u_temp = up  
    v_temp = vp  
    x = rho_R(u_temp,v_temp,traindata,labels,eps,k)
    return x
    

def rho_R(u,v,traindata,labels,eps,k):
    p=u
    q=v
    st_u = p
    st_v = q
    u_init = u
    print st_u
    print st_v
    
    for i in range(k-1):
        t = 0
        # change range to R
        #u_init = random_integers(-10,10,size=(1,31))
        u_init =np.random.rand(1,31)
        u_init = Gram_Schmidt(np.vstack((st_u,u_init)))
        u_init =u_init[-1]
        dist2 = 0
        while True:
            t = t + 1
            
            cont_u = calc_contraction_u(traindata,u_init)
            v1 = inv(np.dot(cont_u.T,cont_u))
            v2 = np.dot(cont_u.T,labels)
            v_temp = np.dot(v1,v2)
            t_st_v = np.vstack((st_v,v_temp))
            v_gs = Gram_Schmidt(t_st_v)
            v_temp = v_gs[-1]
            vp = v_temp
            #st_v =np.vstack((st_v,v))
            #print vv.shape
            cont_v = calc_contraction_v(traindata,v_temp)
            u1 = inv(np.dot(cont_v.T,cont_v))
            u2 = np.dot(cont_v.T,labels)
            u_temp = np.dot(u1,u2)
            t_st_u = np.vstack((st_u,u_temp))
            u_gs = Gram_Schmidt(t_st_u)
            #print st_u.shape
            u_temp =u_gs[-1]
            u_temp = normalize(u_temp)
            #st_u = np.vstack((st_u,u))
            up = u_temp
            dist1 = np.linalg.norm(u_temp - st_u[-1])
            #dist1 = dist(u_temp , st_u[-1])
            if(abs(dist2 - dist1) < eps):
                break
            u_init = u_temp
            print t, abs(dist2 - dist1)
            dist2 = dist1
            #print t 
            #st_u = u_gs
            #st_v = v_gs
           
        u_temp = up  
        v_temp = vp 
        st_u = np.vstack((st_u,u_temp)) 
        st_v = np.vstack((st_v,v_temp)) 
        print st_u.shape
            #print t
        #print u.shape,v.shape
        #plt.imshow(np.outer(uu, vv.T))
        #plt.show()
        #print i
        #p = u_gs
        #q = v_gs
    xxx= 0
    st_u.dump('u.dat')
    st_v.dump('v.dat')
    for i in range(st_u.shape[0]):
        xxx += np.outer(st_u[i], st_v[i].T)
        
    #xxx[xxx < 0] = 0
    #
    x = normalize((xxx))
    #x = xxx
    #x = 255.0 * xxx/xxx.max()
    ##plt.imshow(x,cmap = plt.get_cmap('gray'))#, cmap = cm.Grays_r)
    ##plt.show()
    scipy.misc.imsave('outfile.jpg', x) 
    #print x 
    #return x
    XM(x,traindata,labels,3.0)
    return x

def XM(M,traindata,labels,threshold):
    n = len(traindata)
    #for i in range(n):
    c_p = 0
    c_n = 0
    
    n1 = len(labels)
    pp1 = np.empty((n1,1))
    #mean = M.mean(axis=0)
    #M = M - mean[np.newaxis,:]
    for i in range(n):
        #print labels
        img = np.reshape(traindata[i], (31,81))
        pp = img*M#np.dot(M,img.T)#,'valid')
        px = (np.mean(pp) - np.var(pp)) * 1000000.0
        pp1[i] = px
        print px
        if(px>threshold):
            c_p += 1
        else:
            c_n += 0
            
    pp1.dump('hist_data.dat')
    print c_p , c_n
    

def predict(testdata, M, threshold):
    n = len(testdata)
    predictions = np.empty((n,1))
    for i in range(n):
        img = np.reshape(testdata[i], (31,81))
        pp = img*M#np.dot(M,img.T)#,'valid')
        px = (np.mean(pp) - np.var(pp)) * 1000000.0
        if (px < threshold):
            predictions[i] = 0
        else:
            predictions[i] = 1
    return predictions

def getW(traindata, labels, eps, k):
    W = rho_1(traindata,labels,eps,k)
    return W

# M = patch
# filename = test image filename(full path)
def slidingWindow(img, M,threshold):
    img = Image.open(img)
    img = np.asarray(img)
    img2 = np.pad(img,(90,90),'constant', constant_values=(0,0))
    val = np.empty((img.shape[0],img.shape[1]))
    for j in range(val.shape[0]):
        for i in range(val.shape[1]):
            temp = img2[j+90:j+121,i+90:i+171]
            val[j,i] = testOnImage(M,temp,threshold)
            #print val[j,i]
    
    val = scipy.ndimage.binary_erosion(val).astype(val.dtype)
    val = np.where(val == val.max())
    #val = np.asarray(val)
    a = val[0]
    b= val[1]
    #ij = np.unravel_index(np.argmax(val), val.shape)
    #x, y = ij[::-1]
       #  #fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(8, 3))
   #  
   #  
    plt.imshow(img,cmap ="Greys_r")
    ct = plt.gca()
    ct.set_axis_off()
    ct.set_title('image')
   # highlight matched region
    hcoin = 35
    wcoin = 81
    n = len(a)
    for i in range(n):
        rect = plt.Rectangle((b[i], a[i]), wcoin, hcoin, edgecolor='r', facecolor='none')
        ct.add_patch(rect)
   # 
     #  # highlight matched region
    plt.autoscale(False)
     #  #plt.plot(x, y, 'o', markeredgecolor='r', markerfacecolor='none', markersize=10)
    #  
    plt.show()
    
    #plt.imshow(val)
    #plt.show()
    #plt.imshow(img)
    #plt.show()
    return val
    

def testOnImage(M,img,thresh):
    pp = img*M#np.dot(M,img.T)#,'valid')
    px = (np.mean(pp) - np.var(pp)) * 1000000.0
    if (px > thresh):
        return 1
    else:
        return 0

if __name__ == '__main__':
    traindata = np.load('train_data_new.dat')
    labels = np.load('labels_new.dat')
    #u_init = random.sample(range(256), 81)
    #get_trainingdata()
    
#     print "Normalizing..."
#     mean = traindata.mean(axis=0)
#     traindata = traindata - mean[np.newaxis,:]
    M = rho_1(traindata,labels,0.0001,9)
    #slidingWindow('/Users/abhishek/Documents/workspace/tensor_LDA/Test/TEST_40.PGM',M)
    #print u,v
    #calc_contraction_u(train_)
    
    