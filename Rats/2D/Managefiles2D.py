#Function to read, load and preprocess all the nifti data


import NumpyImage, helpingtools
import os
import nibabel as nib
import numpy as np

class myFiles2D:

    def __init__(self, directoryBrains, directoryClassifiedBrains, numberOfFiles):
        self.directoryBrains = directoryBrains
        self.directoryClassifiedBrains = directoryClassifiedBrains
        self.numberOfFiles=numberOfFiles

    def openlist(self, type):
        filesnumpy=[]
        for i in range(0,self.numberOfFiles):
            if (type=='mri'):
                #helpingtools.update_progress("Loading MRI:",i/self.numberOfFiles)
                files_list = sorted(os.listdir(self.directoryBrains))
                print(len(files_list))
                filepath = self.directoryBrains + "/" + files_list[i]
                numpy = nib.load(filepath)
                print('Beggining to pre-process the brain number' + str(i)+'   ' + files_list[i])
                imagemnumpy=NumpyImage.myNumpy(numpy)
                imagemnumpy=imagemnumpy.preProcessing()
                filesnumpy.append(imagemnumpy)
                #print(imagemnumpy.shape)
            else:
                #helpingtools.update_progress("Loading MASK:",i/self.numberOfFiles)
                files_list = sorted(os.listdir(self.directoryClassifiedBrains))
                filepath = self.directoryClassifiedBrains + "/" + files_list[i]
                numpy = nib.load(filepath)
                print('Saving mask number:   ' + str(i) + '   ' + files_list[i])
                numpyfile = numpy.get_data()
                filesnumpy.append(numpyfile)
                #print(numpyfile.shape)
        if(type=="mri"):
            helpingtools.update_progress("Loading MRI:",1)
        else:
            helpingtools.update_progress("Loading MASK:",1)
        filesnumpy = np.concatenate(filesnumpy, 2)
        filesnumpy = np.rollaxis(filesnumpy, 2).reshape(filesnumpy.shape[2], 64, 64,1 )
        print(filesnumpy.shape)
        return filesnumpy

    def saveFiles(self):
        mris = self.openProcesslist("mri")
        masks = self.openProcesslist("mask")
        #guardar num file para depois usar
        return "ok"

        