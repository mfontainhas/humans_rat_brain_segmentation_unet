#Make the list of numpy arrays to
import os

import NumpyImage
import helpingtools
import nibabel as nib
import numpy as np


class myFiles2D:

    def __init__(self, directoryBrains, numberOfFiles):
        self.directoryBrains = directoryBrains
        self.numberOfFiles = numberOfFiles
        self.shapex = 256
        self.shapey = 256
        self.shapez = 256

    def openlist(self):
        mri = []
        mri_older_train =[]
        mask_older_train=[]
        mri_older_val =[]
        mask_older_val=[]
        mask = []
        no_brain_list=[]
        mri_notzero = []
        mask_notzero = []
        files_list = sorted(os.listdir(self.directoryBrains))
        PATHOLDER="/Notebooks/Marianadissertation/Humans/OLDERTOTRAIN"
        older=sorted(os.listdir(PATHOLDER))
        print (len(older))
        for i in range(0, len(older), 3):
            print(older[i])
            print(older[i+1])
            oldert1 = nib.load(PATHOLDER+"/"+older[i])
            oldermask = nib.load(PATHOLDER+"/"+older[i+1])
            oldermask = oldermask.get_data()
            oldert1 = NumpyImage.myNumpy(oldert1)
            oldert1 = oldert1.preProcessing()
            #oldermask = NumpyImage.myNumpy(oldermask)
            #oldermask = oldermask.preProcessing()
            if i < len(older)-6:
                mri_older_train.append(oldert1)
                mask_older_train.append(oldermask)
            else:
                mri_older_val.append(oldert1)
                mask_older_val.append(oldermask)
        
        if self.numberOfFiles > int(len(files_list) / 7):
            print("You have reached the maximum number of patients. Number of patients is now " + str(
                int(len(files_list) / 7)))
            self.numberOfFiles = int(len(files_list) / 7)
        for i in range(0, self.numberOfFiles * 7, 7):
            helpingtools.update_progress("Loading and Pre-Processing MRI and labelling MASK",
                                         i / (self.numberOfFiles * 7))
            filepatht1 = self.directoryBrains + "/" + files_list[i]
            filepathwmg = self.directoryBrains + "/" + files_list[i + 3]
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
            numpywmg = nib.load(filepathwmg)
            print('Saving mask number:   ' + str(i) + '   ' + files_list[i + 3])
            numpywmg = numpywmg.get_data()
            if (self.shapex != numpywmg.shape[1]):
                numpywmg = np.append(numpywmg,
                                     np.zeros((numpywmg.shape[0], self.shapex - numpywmg.shape[1], numpywmg.shape[2])),
                                     axis=1)
            if (self.shapey != numpywmg.shape[2]):
                numpywmg = np.append(numpywmg,
                                     np.zeros((numpywmg.shape[0], numpywmg.shape[1], self.shapey - numpywmg.shape[2])),
                                     axis=2)
            # print(numpywmg.shape)
            mask.append(numpywmg)
            
        mri=mri_older_train + mri + mri_older_val
        mask=mask_older_train + mask + mask_older_val
        mri = np.concatenate(mri, 2)
        mri = np.rollaxis(mri, 2).reshape(mri.shape[2], mri.shape[0], mri.shape[1], 1)
        print(mri.shape)
        mask = np.concatenate(mask, 2)
        mask = np.rollaxis(mask, 2).reshape(mask.shape[2], mask.shape[0], mask.shape[1], 1)
        print(mask.shape)
        for i in range(len(mask)):
            if np.count_nonzero(mask[i]) != 0:
                mri_notzero.append(mri[i])
                mask_notzero.append(mask[i])
        '''for i in range(len(mask)):
            if (np.max(mask[i]) == 0):
                no_brain_list.append(i)
                print(i)
        mask = np.delete(mask, no_brain_list, axis=0)
        mri = np.delete(mri, no_brain_list, axis=0)'''
        mri=np.asarray(mri_notzero)
        mask=np.asarray(mask_notzero)
        print(mri.shape)
        print(mask.shape)

        helpingtools.update_progress("Loading and Pre-Processing MRI and labelling MASK", 1)
        # print(mask_labble.shape)
        return mri, mask


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
