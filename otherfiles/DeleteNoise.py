import cv2
import nibabel as nib
from scipy.misc import toimage
import os
import random
import numpy as np


numpy=[]
ONE_IMAGE='/home/mariana/Desktop/teens_binaseg'
def runAll(path):
    files_list = sorted(os.listdir(path))
    for i in range(len(files_list)):
        filepath = path + "/" + files_list[i]
        #filepath=ONE_IMAGE
        print(files_list[i])
        s = files_list[i].split('_')

        numpya = nib.load(filepath)
        numpy = numpya.get_data()
        #numpy = preparemask(numpy)
        #numpy = np.rollaxis(numpy, 2)
        #numpy[numpy > 0] = 1
        numpy[numpy >= 0.2] = 1
        numpy[numpy < 0.2] = 0
        #toimage(numpy[130]).show()#
        #mask=preparemask(numpy)
        #mask = cleanmask(mask)
        #toimage(mask[130]).show()
        #mask = np.rollaxis(mask, 2)
        #mask = np.rollaxis(mask, 2)
        img = nib.Nifti1Image(numpy, None)
        img.to_filename(filepath)
        #+s[0] + "_" + s[1] + "_" + s[2] +



def preparemask(la):
    mask=la
    for j in range(len(la[0])):
        blur = cv2.GaussianBlur(la[j], (5, 5), 0)
        __, th3 = cv2.threshold(blur, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        (__, cnts, _) = cv2.findContours(th3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(th3, cnts,-1, 1,-1)
        mask[j] = th3
    return mask

def cleanmask(la):
    mask=la
    a=0
    v=0
    for j in range(len(la[0])):
        blur = cv2.GaussianBlur(la[j], (5, 5), 0)
        __, th3 = cv2.threshold(blur, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        (__, cnts, _) = cv2.findContours(th3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        #la[j] = th3
        listcountours = []
        numerocontornos = 0
        for c in cnts:
            area = cv2.contourArea(c)
            listcountours.append(area)
            numerocontornos = numerocontornos + 1
            if numerocontornos==0:
                mask[j]=th3
                v=v+1
            else:
                print("entrei")
                a=a+1
                maximo = max(listcountours)
                if maximo >700:
                    print(maximo)
                    index = indices(listcountours, maximo)
                    numero = random.choice(index)
                    print(numero)
                    th3=cv2.drawContours(th3, cnts[numero],-1, 255,-1)
                    mask[j]=th3
    print(a)
    print(v)
    return mask

def fillblack(numpy):
    print(numpy)
    for i in range(numpy.shape[0]):
        for j in range(numpy.shape[1]):
            min_black=300
            max_black=0
            if(numpy[i][j]==255):
                if(min_black>j):
                    min_black=j
                elif(max_black<j):
                    max_black=j
            for i in range(max_black-min_black): #pintar os que estao entre min e max
                numpy[i][j]=0
    return numpy

def indices( mylist, value):
    return [i for i,x in enumerate(mylist) if x==value]

if __name__ == '__main__':
    runAll(ONE_IMAGE)