import os

import NumpyImage
import helpingtools
import nibabel as nib
import numpy as np
from keras import utils

class myFiles2D:

    def __init__(self, directoryBrains, numberOfFiles,CLASS_NUMBER):
        self.directoryBrains = directoryBrains
        self.numberOfFiles = numberOfFiles
        self.shapex = 256
        self.shapey = 256
        self.shapez = 256
        self.CLASS_NUMBER=CLASS_NUMBER

    def openlist(self):
        mri = []
        mask = []
        wmg = []
        mri_notzero = []
        wmg_notzero = []
        files_list = sorted(os.listdir(self.directoryBrains))
        if self.numberOfFiles > int(len(files_list) / 7):
            print("You have reached the maximum number of patients. Number of patients is now " + str(
                int(len(files_list) / 7)))
            self.numberOfFiles = int(len(files_list) / 7)
        for i in range(0, self.numberOfFiles * 7, 7):
            helpingtools.update_progress("Loading and Pre-Processing MRI and labelling MASK",
                                         i / (self.numberOfFiles * 7))
            filepatht1 = self.directoryBrains + "/" + files_list[i]
            filepathwmg = self.directoryBrains + "/" + files_list[i + 6]
            filepathmask = self.directoryBrains + "/" + files_list[i + 3]
            # Guardar T1 na lista mris
            print('Pre-process the brain number:   ' + str(i) + '   ' + files_list[i])
            numpyt1 = nib.load(filepatht1)
            numpyt1 = NumpyImage.myNumpy(numpyt1)
            numpyt1 = numpyt1.preProcessing()
            if (self.shapex != numpyt1.shape[1]):
                numpyt1 = np.append(numpyt1,
                                    np.zeros((numpyt1.shape[0], self.shapex - numpyt1.shape[1], numpyt1.shape[2])),
                                    axis=1)
            if (self.shapey != numpyt1.shape[2]):
                numpyt1 = np.append(numpyt1,
                                    np.zeros((numpyt1.shape[0], numpyt1.shape[1], self.shapey - numpyt1.shape[2])),
                                    axis=2)
            if (self.shapez != numpyt1.shape[0]):
                numpyt1 = np.append(numpyt1,
                                    np.zeros((self.shapez - numpyt1.shape[0], numpyt1.shape[1], numpyt1.shape[2])),
                                    axis=0)
            mri.append(numpyt1)
            # Save brain_seg in Masks
            numpymask = nib.load(filepathmask)
            print('Saving mask number:   ' + str(i) + '   ' + files_list[i + 3])
            numpymask = numpymask.get_data()
            if (self.shapex != numpymask.shape[1]):
                numpymask = np.append(numpymask, np.zeros(
                    (numpymask.shape[0], self.shapex - numpymask.shape[1], numpymask.shape[2])), axis=1)
            if (self.shapey != numpymask.shape[2]):
                numpymask = np.append(numpymask, np.zeros(
                    (numpymask.shape[0], numpymask.shape[1], self.shapey - numpymask.shape[2])), axis=2)
            mask.append(numpymask)
            # Segment brain
            # prepare wmg images
            numpywmg = nib.load(filepathwmg)
            print('Saving wmg number:   ' + str(i) + '   ' + files_list[i + 6])
            numpywmg = numpywmg.get_data()
            if (self.shapex != numpywmg.shape[1]):
                numpywmg = np.append(numpywmg,
                                     np.zeros((numpywmg.shape[0], self.shapex - numpywmg.shape[1], numpywmg.shape[2])),
                                     axis=1)
            if (self.shapey != numpywmg.shape[2]):
                numpywmg = np.append(numpywmg,
                                     np.zeros((numpywmg.shape[0], numpywmg.shape[1], self.shapey - numpywmg.shape[2])),
                                     axis=2)
            wmg.append(numpywmg)

        mri = self.concat_roll(mri)
        mask = self.concat_roll(mask)
        #mask=np.multiply(mask,1/255)
        print(np.max(mask))
        #Set to zero all non-brain pixels (following the pipeline rules) SEGMENTATION BRAIN
        mri = np.multiply(mri, mask)
        wmg = self.concat_roll(wmg)
        #Remove all non-brain layers (following the pipeline rules) CLASSIFICATION LAYER
        for i in range(0, mask.shape[0]):
            if np.count_nonzero(mask[i]) != 0:
                mri_notzero.append(mri[i])
                wmg_notzero.append(wmg[i])

        mri_notzero = np.asarray(mri_notzero)
        #print(mri_notzero.shape)
        wmg_notzero = np.asarray(wmg_notzero)
       # print("Multi-Classifier Shape after removing Non-Brain Layers: "+str(wmg_notzero.shape))
        print(np.max(wmg_notzero))
        helpingtools.update_progress("Loading and Pre-Processing MRI and labelling MASK", 1)
        wmg_notzero = utils.to_categorical(wmg_notzero, self.CLASS_NUMBER)
        print(wmg_notzero.shape)
        return mri_notzero, wmg_notzero

    def concat_roll(self, numpy):
        mri = np.concatenate(numpy, 2)
        mri = np.rollaxis(mri, 2).reshape(mri.shape[2], mri.shape[0], mri.shape[1], 1)
        print(mri.shape)
        return mri


''' def openlist(self, type):
        filesnumpy=[]
        for i in range(2,self.numberOfFiles):
            if (type=='mri'):
                helpingtools.update_progress("Loading MRI:",i/self.numberOfFiles)
                files_list = sorted(os.listdir(self.directoryBrains))
                filepath = self.directoryBrains + "/" + files_list[i]
                numpy = nib.load(filepath)
                #print('Beggining to pre-process the brain number' + str(i)+'   ' + files_list[i])
                imagemnumpy=NumpyImage.myNumpy(numpy)
                imagemnumpy=imagemnumpy.preProcessing()
                filesnumpy.append(imagemnumpy)
                #print(imagemnumpy.shape)
            else:
                helpingtools.update_progress("Loading MRI:",i/self.numberOfFiles)
                files_list = sorted(os.listdir(self.directoryClassifiedBrains))
                filepath = self.directoryClassifiedBrains + "/" + files_list[i]
                numpy = nib.load(filepath)
                #print('Saving mask number:   ' + str(i) + '   ' + files_list[i])
                numpyfile = numpy.get_data()
                filesnumpy.append(numpyfile)
                #print(numpyfile.shape)
        if(type=="mri"):
            helpingtools.update_progress("Loading MRI:",1)
        else:
            helpingtools.update_progress("Loading MASK:",1)
        for i in range(len(filesnumpy)):
            if 
            
        filesnumpy = np.concatenate(filesnumpy, 2)
        filesnumpy = np.rollaxis(filesnumpy, 2).reshape(filesnumpy.shape[2], 64, 64,1 )
        print(filesnumpy.shape)
        return filesnumpy

    def saveFiles(self):
        mris = self.openProcesslist("mri")
        masks = self.openProcesslist("mask")
        #guardar num file para depois usar
        return "ok"'''
