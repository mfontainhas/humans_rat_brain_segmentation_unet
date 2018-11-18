#Make the list of numpy arrays to
import NumpyImage, helpingtools
import os
import nibabel as nib
import numpy as np
from keras import utils

class myFiles3D:

    def __init__(self, directoryBrains, directoryClassifiedBrains, directoryBinaryMask):
        self.directoryBrains = directoryBrains
        self.directoryClassifiedBrains = directoryClassifiedBrains
        self.directoryBinaryMask=directoryBinaryMask

    def openlist(self):
        filesnumpymri=[]
        filesnumpymask=[]
        files_list_MRI = (os.listdir(self.directoryBrains))
        print(files_list_MRI)
        for i in range(0,144):
            #helpingtools.update_progress("Loading Images:",i)
            filesmaskbinary=sorted(os.listdir(self.directoryBinaryMask))
            numpymri=nib.load(self.directoryBrains+"/"+files_list_MRI[i])
            splits_MRI=files_list_MRI[i].split("_")
            nameMRI=splits_MRI[1]+"_"+splits_MRI[2]+"_"+splits_MRI[3]
            for namebinary in filesmaskbinary:
                if nameMRI in namebinary:
                    numpymaskbinary=nib.load(self.directoryBinaryMask+"/"+namebinary)
                    numpymaskbinary=numpymaskbinary.get_data()
                    imagemnumpy=NumpyImage.myNumpy(numpymri)
                    imagemnumpy=imagemnumpy.preProcessing()
                    filesnumpymri.append(imagemnumpy)
                    filesnumpymask.append(numpymaskbinary)
        
        filesnumpymri=np.asarray(filesnumpymri)
        print(filesnumpymri.shape)
        filesnumpymri = np.reshape(filesnumpymri,[filesnumpymri.shape[0], 64, 64,40,1 ])
        print(filesnumpymri.shape)
        filesnumpymask=np.asarray(filesnumpymask)
        filesnumpymask = np.reshape(filesnumpymask,[filesnumpymask.shape[0], 64, 64,40,1 ])
        #random shuffle all layers
        #helpingtools.update_progress("Loading Images:",1)
        return filesnumpymri, filesnumpymask
            
            
    def segmentBrain(self,numpybrain,numpymask): #put 0 values in non brain voxel segmented brain
        return np.multiply(numpybrain,numpymask)
        
    def random_shuffle(self,mris,masks):
        #len=5520
        idx = np.random.permutation(len(mris))
        mris,masks = mris[idx], masks[idx]
        print(idx)
        return mris,masks