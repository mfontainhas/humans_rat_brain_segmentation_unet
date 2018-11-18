import numpy as np
from skimage import exposure

class myNumpy:
    def __init__(self,numpyfile):
        self.numpyfile=numpyfile

    def preProcessing(self):
        imagemnumpy = (self.numpyfile).get_data()
        #NORMALIZATION
        numpynormalized = np.floor(imagemnumpy)
        numpynormalized /= np.max(numpynormalized)  # divisao pelo max   
        #HISTOGRAM EQ
        #for i in range(len(imagemnumpy)):
            #numpynormalized[i]=exposure.equalize_adapthist(numpynormalized[i],clip_limit=0.3)
        #NORMALIZATION
       # numpynormalized = np.floor(numpynormalized)
       # numpynormalized /= np.max(numpynormalized)  # divisao pelo max   
        
        return numpynormalized
#for i in range(len(imagemnumpy)):
#            numpynormalized[i]=exposure.equalize_adapthist(numpynormalized[i],clip_limit=0.01)