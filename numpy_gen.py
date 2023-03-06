import numpy as np

def Mate(a,b):
    res=np.empty(a.shape)
    raded=np.random.rand(*a.shape)>0.5
    raded2=np.logical_not(raded)
    res[raded]=a[raded]
    res[raded2]=b[raded2]
    return res
def Mute(a,e=0.1):
    raded=np.random.rand(*a.shape)<e
    r = np.random.rand(*a.shape)
    a[raded]=r[raded]
    return a
def relu(inX):
    return np.maximum(0,inX)
def softmax(x):
    ep=np.exp(x)
    return ep/np.sum(ep)
def sigmoid(x):
    return 1/(1 + np.exp(-x))
if __name__=="__main__":
    # input array
    x = np.array([[ 0.1, 0.2, 0.3], [ 0.4, 0.5, 0.6], [ 0.7, 0.8, .9]])
    y = np.array([[ 0.11, 0.22, 0.33], [ 0.44, 0.55, 0.66], [ 0.77, 0.88, 0.99]])
    res=Mate(x,y)
    print(res)
    res=Mute(res)
    print(sigmoid(res))