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
        files_list_MASKC=sorted(os.listdir(self.directoryClassifiedBrains))
        for i in range(0,144):
            helpingtools.update_progress("Loading Images:",i/len(files_list_MASKC))
            splits_MASKC=files_list_MASKC[i].split("_")
            nameMASKC=splits_MASKC[1]+"_"+splits_MASKC[2]+"_"+splits_MASKC[3]
            for j in range(1,len(files_list_MRI)):
                splits_MRI=files_list_MRI[j].split("_")
                nameMRI=splits_MRI[1]+"_"+splits_MRI[2]+"_"+splits_MRI[3]
                if nameMASKC==nameMRI:
                    filesmaskbinary=sorted(os.listdir(self.directoryBinaryMask))
                    numpymri=nib.load(self.directoryBrains+"/"+files_list_MRI[j])
                    numpymask=(nib.load(self.directoryClassifiedBrains+"/"+files_list_MASKC[i])).get_data()
                    numpymaskbinary=nib.load(self.directoryBinaryMask+"/Brain_Mask_"+nameMRI+".nii")
                    numpymaskbinary=numpymaskbinary.get_data()
                    imagemnumpy=NumpyImage.myNumpy(numpymri)
                    imagemnumpy=imagemnumpy.preProcessing()
                    #numpymaskbinary=NumpyImage.myNumpy(numpymaskbinary)
                    #numpymaskbinary=numpymaskbinary.preProcessing()
                    imagemnumpy=self.segmentBrain(imagemnumpy,numpymaskbinary)
                    filesnumpymri.append(imagemnumpy)
                    filesnumpymask.append(numpymask)
        #filesnumpymri,filesnumpymask=self.random_shuffle(filesnumpymri,filesnumpymask)
        #filesnumpymri = np.concatenate(filesnumpymri, 2)
        filesnumpymri=np.asarray(filesnumpymri)
        print(filesnumpymri.shape)
        filesnumpymri = np.reshape(filesnumpymri,[filesnumpymri.shape[0], 64, 64,40,1 ])
        print(filesnumpymri.shape)
        filesnumpymask=np.asarray(filesnumpymask)
        filesnumpymask = np.reshape(filesnumpymask,[filesnumpymask.shape[0], 64, 64,40,1 ])
        #random shuffle all layers
        filesnumpymask = utils.to_categorical(filesnumpymask, 4)
        print(filesnumpymask.shape)
        print(filesnumpymri.shape)
        helpingtools.update_progress("Loading Images:",1)
        return filesnumpymri, filesnumpymask
            
            
    def segmentBrain(self,numpybrain,numpymask): #put 0 values in non brain voxel segmented brain
        return np.multiply(numpybrain,numpymask)
        
