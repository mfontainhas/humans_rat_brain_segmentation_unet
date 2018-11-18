import os
import nibabel as nib
import cv2
import numpy as np

def prepareWGM(path):
    files_list = sorted(os.listdir(path))
    for number in range(0,len(files_list)):
        filepath = path + "/" + files_list[number]
        print(files_list[number])
        s=files_list[number].split('_')
        print(s[0])
        if s[1]=='aseg.nii.gz':
            numpya = nib.load(filepath)
            tamanho=numpya.shape
            numpy = numpya.get_data()

            for i in range(tamanho[0]):
                print("Transforming     :  " + str((i / 256) * 100) + "%")
                for j in range(tamanho[1]):
                    for z in range(tamanho[2]):
                        if numpy[i][j][z]!=0:
                            if numpy[i][j][z]==41 or numpy[i][j][z]==2: #white
                                numpy[i][j][z]=1
                            elif numpy[i][j][z]==42 or numpy[i][j][z]==3: #gray
                                numpy[i][j][z]=2
                            else:
                                numpy[i][j][z]=0
            img = nib.Nifti1Image(numpy,None)
            img.to_filename(path+"/"+s[0]+"_wgm.nii.gz")

def prepareSubC(path):
    files_list = sorted(os.listdir(path))
    for a in range(0,len(files_list)):
        filepath = path + "/" + files_list[a]
        print(files_list[a])
        s=files_list[a]
        print(s[:-12])
        if s[-11:]=='aseg.nii.gz':
            numpya = nib.load(filepath)
            tamanho=numpya.shape
            numpy = numpya.get_data()
            for i in range(tamanho[0]):
                print("Transforming     :  " + str((i / 256) * 100) + "%")
                for j in range(tamanho[1]):
                    for z in range(tamanho[2]):
                        if numpy[i][j][z]!=0:
                            if numpy[i][j][z]==42 or numpy[i][j][z]==3: #gray
                                numpy[i][j][z] = 2
                            elif numpy[i][j][z]==41 or numpy[i][j][z]==2: #white
                                numpy[i][j][z]=1
                            elif numpy[i][j][z] == 47 or numpy[i][j][z] == 46 or numpy[i][j][z] == 8 or numpy[i][j][z] == 7:  # CEREBRELLUM
                                numpy[i][j][z] = 3
                            elif numpy[i][j][z] == 10 or numpy[i][j][z] == 49 or numpy[i][j][z] == 50 or numpy[i][j][z] == 11 or numpy[i][j][z] == 52 or numpy[i][j][z] == 13:  # SUBC
                                numpy[i][j][z] = 4
                            else:
                                numpy[i][j][z]=0
            img = nib.Nifti1Image(numpy,None)
            img.to_filename("/home/mariana/Desktop/wgmscteens/"+files_list[a])


def preparemask(path):
    files_list = sorted(os.listdir(path))
    for i in range(len(files_list)):
        filepath = path + "/" + files_list[i]
        print(files_list[i])
        s=files_list[i].split('_')
        numpya = nib.load(filepath)
        tamanho=numpya.shape
        numpy = numpya.get_data()
        newnumpy=[]

        for i in range(tamanho[0]):
            print("Transforming     :  " + str((i / 256) * 100) + "%")
            for j in range(tamanho[1]):
                for z in range(tamanho[2]):
                    if numpy[i][j][z]>0: #mask
                        numpy[i][j][z]=1

        img = nib.Nifti1Image(numpy,None)
        img.to_filename(filepath)

def preparemask2(path):
    files_list = sorted(os.listdir(path))
    for i in range(len(files_list)):
        filepath = path + "/" + files_list[i]
        print(files_list[i])
        s = files_list[i].split('_')
        numpya = nib.load(filepath)
        numpy = numpya.get_data()
        for j in range(numpy.shape[0]):
            # global thresholding
            blur = cv2.GaussianBlur(numpy[j], (5, 5), 0)
            __, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            _, contours, _ = cv2.findContours(th3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(th3, contours, -1, 255, -1)
            numpy[j]=th3


        img = nib.Nifti1Image(numpy, None)
        img.to_filename(filepath)
def invert(numpy):

    return numpy
if __name__ == '__main__':
    prepareWGM('/Notebooks/Marianadissertation/Humans/TestImagesTeens')
   #prepareSubC('/home/mariana/Desktop/wgmscteens')