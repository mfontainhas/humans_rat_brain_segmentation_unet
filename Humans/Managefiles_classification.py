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
        mask = []
        numero1 = 0
        numero0 = 0
        files_list = sorted(os.listdir(self.directoryBrains))
        if self.numberOfFiles > int(len(files_list) / 6):
            print("You have reached the maximum number of patients. Number of patients is now " + str(
                int(len(files_list) / 6)))
            self.numberOfFiles = int(len(files_list) / 6)
        for i in range(0, self.numberOfFiles * 6, 6):
            helpingtools.update_progress("Loading and Pre-Processing MRI and labelling MASK",
                                         i / (self.numberOfFiles * 6))
            filepatht1 = self.directoryBrains + "/" + files_list[i]
            filepathwmg = self.directoryBrains + "/" + files_list[i + 3]
            # Guardar T1 na lista mris
            #print('Pre-process the brain number:   ' + str(i)+'   ' + files_list[i])
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
            #print('Saving mask number:   ' + str(i) + '   ' + files_list[i+2])
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

        mask_labels = []
        mask = np.concatenate(mask, 2)
        mask = np.rollaxis(mask, 2).reshape(mask.shape[2], mask.shape[0], mask.shape[1], 1)
        '''for i in range(0, mask.shape[0]):
            if np.count_nonzero(mask[i, :, :, 0]) == 0:
                mask_labels.append(0)
                numero0 = numero0 + 1
            else:
                mask_labels.append(1)
                numero1 = numero1 + 1'''
        for i in range(len(masks)):
            if (np.max(masks[i]) == 0):
                mask_labels.append(0)
                no_brain_list.append(i)
                numero0 = numero0 + 1
            else:
                mask_labels.append(1)
                numero1 = numero1 + 1 
        mri = np.concatenate(mri, 2)
        mri = np.rollaxis(mri, 2).reshape(mri.shape[2], mri.shape[0], mri.shape[1], 1)
        mask_labels = np.reshape(mask_labels, [mri.shape[0], 1])
        helpingtools.update_progress("Loading and Pre-Processing MRI and labelling MASK", 1)
        print("\n------------  BRAIN READY FOR LAYER CLASSIFICATION  ------------")
        print("|            Number of images with BRAIN: " + str(numero1) + "                |")
        print("|          Number of images without BRAIN: " + str(numero0) + "               |")
        print("---------------------------------------------------------------")
        print(mask_labels.shape)
        return mri, mask_labels

    def saveFiles(self):
        mris = self.openProcesslist("mri")
        masks = self.openProcesslist("mask")
        # guardar num file para depois usar
        return "ok"
