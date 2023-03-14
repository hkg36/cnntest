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
    r = np.random.rand(*a.shape)*0.01-0.005
    a[raded]+=r[raded]
    return a
def relu(inX):
    return np.maximum(0,inX)
def leakyrelu(inX):
    return np.maximum(inX*0.1,inX)
def softmax(x):
    ep=np.exp(x)
    return ep/np.sum(ep)
def sigmoid(x):
    return 1/(1 + np.exp(-x))

class GenNNBase(object):
    def __init__(self) -> None:
        self.score=np.nan
    def forward(self, x):
        tmp=x
        for l in self.line:
            if l[0]=="func":
                tmp=l[1](tmp)
            elif l[0]=="array":
                tmp=np.dot(tmp,l[1])
        return tmp
    def Mute(self,per=0.05):
        for l in self.line:
            if l[0]=="array":
                Mute(l[1],per)
class GenNNFactory(object):
    def __init__(self,*argv) -> None:
        self.nnCls=GenNNBase
        self.line=[]
        for o in argv:
            if callable(o):
                self.line.append(["func",o])
            elif isinstance(o,(list,tuple)):
                self.line.append(["array",o])
    def setNNClass(self,cls=GenNNBase):
        assert issubclass(cls, GenNNBase)
        self.nnCls=cls
    def NewNN(self):
        newone=self.nnCls()
        liner=[]
        for l in self.line:
            if l[0]=="func":
                liner.append(["func",l[1]])
            elif l[0]=="array":
                liner.append(["array",np.random.rand(*l[1])])
        newone.line=liner
        return newone
    def Mate(self,na,nb):
        newone=self.nnCls()
        liner=[]
        for i in range(len(self.line)):
            l=self.line[i]
            if l[0]=="func":
                liner.append(["func",l[1]])
            elif l[0]=="array":
                liner.append(["array",Mate(na.line[i][1],nb.line[i][1])])
        newone.line=liner
        return newone
    def Mute(self,na,per=0.05):
        na.Mute(per)
def BuildGenNNFactory(*argv):
    last_layer=np.nan
    lays=[]
    for o in argv:
        if isinstance(o,int):
            if np.isnan(last_layer):
                last_layer=o
            else:
                lays.append([last_layer,o])
                last_layer=o
        elif callable(o):
            lays.append(o)
    return GenNNFactory(*lays)
if __name__=="__main__":
    # input array
    x = np.array([[ 0.1, 0.2, 0.3], [ 0.4, 0.5, 0.6], [ 0.7, 0.8, .9]])
    y = np.array([[ 0.11, 0.22, 0.33], [ 0.44, 0.55, 0.66], [ 0.77, 0.88, 0.99]])
    res=Mate(x,y)
    print(res)
    res=Mute(res)
    print(sigmoid(res))