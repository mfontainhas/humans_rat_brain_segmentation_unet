########################################################################
##  This function is part of the Rat Brain Segmentation DL workflow   ##
########################################################################
# Class to read, load and pre-process all nifti files
# Files will all be handled in full 3D
# Pre-processing includes only the intensity range normalization
########################################################################
# Mandatory inputs include directories for the original MRI brain images, ...
# ... for the images with classified tissues and the binary brain masks.
# Each subject must have the three different images.
########################################################################
# Created by M. Rodrigues, November 2018
# Edited by R. Magalhaes, January 2019
########################################################################

import NumpyImage, helpingtools
import os
import nibabel as nib
import numpy as np
from keras import utils

class myFiles3D:

    def __init__(self, directoryBrains, directoryClassifiedBrains, directoryBinaryMask): #Must by default initiate with directories for 
        self.directoryBrains = directoryBrains                                           #original brain, classified and binary masks images
        self.directoryClassifiedBrains = directoryClassifiedBrains
        self.directoryBinaryMask=directoryBinaryMask

    def openlist(self):
        filesnumpymri=[]      #Variable to save all MRI images for both original MRI and binary masks
        filesnumpymask=[]     #These are the returned variables
        files_list_MRI = (os.listdir(self.directoryBrains))
        print(files_list_MRI)
        for i in range(0,144): #144 because it's the number of subjects we have. Should be changed to be adaptive to input or context.
            #helpingtools.update_progress("Loading Images:",i)
            filesmaskbinary=sorted(os.listdir(self.directoryBinaryMask)) #Can be outside the for cycle right? Doesnt need to be repeated each time
            numpymri=nib.load(self.directoryBrains+"/"+files_list_MRI[i])
            splits_MRI=files_list_MRI[i].split("_") 
            nameMRI=splits_MRI[1]+"_"+splits_MRI[2]+"_"+splits_MRI[3] #Removing garbage from the name keeping only the ID
            for namebinary in filesmaskbinary: #Search through the list of masks for the matching mask file for each subject
                if nameMRI in namebinary:      #If we find the mathcing file
                    numpymaskbinary=nib.load(self.directoryBinaryMask+"/"+namebinary)
                    numpymaskbinary=numpymaskbinary.get_data() #Load it
                    imagemnumpy=NumpyImage.myNumpy(numpymri)
                    imagemnumpy=imagemnumpy.preProcessing() #Pre-process the origninal MRI file
                    filesnumpymri.append(imagemnumpy)       #Append both files to the list
                    filesnumpymask.append(numpymaskbinary)
        
        filesnumpymri=np.asarray(filesnumpymri)
        print(filesnumpymri.shape)                                                       #Check dimensions of matrix
        filesnumpymri = np.reshape(filesnumpymri,[filesnumpymri.shape[0], 64, 64,40,1 ]) #and force the dimensions of each to the expected 3D format
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