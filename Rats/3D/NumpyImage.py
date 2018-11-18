import numpy as np


class myNumpy:
    def __init__(self,numpyfile):
        self.numpyfile=numpyfile

    def preProcessing(self):
        imagemnumpy = (self.numpyfile).get_data()
        #NORMALIZATION
        numpynormalized = np.floor(imagemnumpy)
        numpynormalized /= np.max(numpynormalized)  # divisao pelo max
        
        
        
        return numpynormalized
