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
        files_list = sorted(os.listdir(self.directoryBrains))
        if self.numberOfFiles > int(len(files_list) / 4):
            print("You have reached the maximum number of patients. Number of patients is now " + str(
                int(len(files_list) / 4)))
            self.numberOfFiles = int(len(files_list) / 4)
        for i in range(0, self.numberOfFiles * 4, 4):
            helpingtools.update_progress("Loading and Pre-Processing MRI and MASK", i / (self.numberOfFiles * 4))
            filepatht1 = self.directoryBrains + "/" + files_list[i]
            filepathwmg = self.directoryBrains + "/" + files_list[i + 2]
            # Guardar T1 na lista mris
            # print('Pre-process the brain number:   ' + str(i)+'   ' + files_list[i])
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
            # print(numpyt1.shape)
            if (self.shapez != numpyt1.shape[0]):
                numpyt1 = np.append(numpyt1,
                                    np.zeros((self.shapez - numpyt1.shape[0], numpyt1.shape[1], numpyt1.shape[2])),
                                    axis=0)
            # print(numpyt1.shape)
            mri.append(numpyt1)
            # Save brain_seg in Masks
            numpywmg = nib.load(filepathwmg)
            # print('Saving mask number:   ' + str(i) + '   ' + files_list[i+2])
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

        mri = np.concatenate(mri, 2)
        mask = np.concatenate(mask, 2)
        print(mri.shape)
        print(mask.shape)
        mri = np.rollaxis(mri, 2).reshape(mri.shape[2], mri.shape[0], mri.shape[1], 1)
        mask = np.rollaxis(mask, 2).reshape(mask.shape[2], mask.shape[0], mask.shape[1], 1)
        return mri, mask

    def saveFiles(self):
        mris = self.openProcesslist("mri")
        masks = self.openProcesslist("mask")
        # guardar num file para depois usar
        return "ok"

    def random_shuffle(self, mris, masks):
        # len=5520
        idx = np.random.permutation(len(mris))
        mris, masks = mris[idx], masks[idx]
        print(idx)
        return mris, masks
