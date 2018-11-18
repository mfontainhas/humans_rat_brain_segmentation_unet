import numpy as np


class myNumpy:
    def __init__(self,numpyfile):
        self.numpyfile=numpyfile
        self.shapex=256
        self.shapey=256
        self.shapez=256

    def preProcessing(self):
        imagemnumpy = (self.numpyfile).get_data()
        #NORMALIZATION
        numpynormalized = np.floor(imagemnumpy)
        numpynormalized /= np.max(numpynormalized)  # divisao pelo max
        numpyt1=numpynormalized
        if (self.shapex != numpyt1.shape[1]):
            numpyt1 = np.append(numpyt1,np.zeros((numpyt1.shape[0], self.shapex - numpyt1.shape[1], numpyt1.shape[2])), axis=1)
        if (self.shapey != numpyt1.shape[2]):
            numpyt1 = np.append(numpyt1,np.zeros((numpyt1.shape[0], numpyt1.shape[1], self.shapey - numpyt1.shape[2])),axis=2)
        if (self.shapez != numpyt1.shape[0]):
            numpyt1 = np.append(numpyt1,
            np.zeros((self.shapez - numpyt1.shape[0], numpyt1.shape[1], numpyt1.shape[2])),axis=0)
        return numpyt1
