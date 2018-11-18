#Make the list of numpy arrays to
import NumpyImage, helpingtools
import os
import nibabel as nib
import numpy as np
from keras import utils

class myFiles2D:

    def __init__(self, directoryBrains, directoryClassifiedBrains, directoryBinaryMask,numberOfFiles):
        self.directoryBrains = directoryBrains
        self.directoryClassifiedBrains = directoryClassifiedBrains
        self.numberOfFiles=numberOfFiles
        self.directoryBinaryMask=directoryBinaryMask

    def openlist(self):
        filesnumpymri=[]
        filesnumpymask=[]
        numero=0
        files_list_MRI = (os.listdir(self.directoryBrains))
        files_list_MASKC=sorted(os.listdir(self.directoryClassifiedBrains))
        print(int(len(files_list_MASKC)*0.8))
        print(len(files_list_MASKC))
        for i in range(1,int(len(files_list_MASKC))):
            helpingtools.update_progress("Loading Images:",i/len(files_list_MASKC))
            splits_MASKC=files_list_MASKC[i].split("_")
            nameMASKC=splits_MASKC[1]+"_"+splits_MASKC[2]+"_"+splits_MASKC[3]
            for j in range(0,len(files_list_MRI)):
                splits_MRI=files_list_MRI[j].split("_")
                nameMRI=splits_MRI[1]+"_"+splits_MRI[2]+"_"+splits_MRI[3]
                if nameMASKC==nameMRI:
                    numero=numero+1
                    if numero>110:
                        break
                    print("NUMERO TOTAL:"+str(numero))
                    filesmaskbinary=sorted(os.listdir(self.directoryBinaryMask))
                    numpymri=nib.load(self.directoryBrains+"/"+files_list_MRI[j])
                    numpymask=(nib.load(self.directoryClassifiedBrains+"/"+files_list_MASKC[i])).get_data()
                    numpymaskbinary=nib.load(self.directoryBinaryMask+"/Brain_Mask_"+nameMRI+".nii")
                    print(files_list_MASKC[i])
                    print(files_list_MRI[j])
                    numpymaskbinary = numpymaskbinary.get_data()
                    print(np.max(numpymaskbinary))
                    imagemnumpy=NumpyImage.myNumpy(numpymri)
                    imagemnumpy=imagemnumpy.preProcessing()
                    imagemnumpy=self.segmentBrain(imagemnumpy,numpymaskbinary)
                    filesnumpymri.append(imagemnumpy)
                    filesnumpymask.append(numpymask)      
        print("NUMERO TOTAL:"+str(numero))
        filesnumpymri = np.concatenate(filesnumpymri, 2)
       # print (filesnumpymri.shape)
        filesnumpymri = np.rollaxis(filesnumpymri, 2).reshape(filesnumpymri.shape[2], 64, 64,1 )
        filesnumpymask = np.concatenate(filesnumpymask, 2)

        filesnumpymask = np.rollaxis(filesnumpymask, 2).reshape(filesnumpymask.shape[2], 64, 64,1 )
        #random shuffle all layers
        filesnumpymri,filesnumpymask=self.random_shuffle(filesnumpymri,filesnumpymask)
        filesnumpymask = utils.to_categorical(filesnumpymask, 4)
        helpingtools.update_progress("Loading Images:",1)
        
        return filesnumpymri, filesnumpymask
            
            
    def segmentBrain(self,numpybrain,numpymask): #put 0 values in non brain voxel segmented brain
        return np.multiply(numpybrain,numpymask)
        
    def random_shuffle(self,mris,masks):
        #len=5520
        idx = np.random.permutation(len(mris))
        mris,masks = mris[idx], masks[idx]
        print(idx)
        return mris,masks