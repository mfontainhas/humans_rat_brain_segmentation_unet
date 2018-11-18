import xlsxwriter
import os
import nibabel as nib
import numpy as np

def putinfoinARRAY(PATHGEN):
    gen_list = sorted(os.listdir(PATHGEN))
    a=0
    array=[]
    for namegen in gen_list: 
        numero_camada=0

        gen_mask=nib.load(PATHGEN+"/"+namegen)
        #gt_mask = gt_mask.get_data()
        gen_mask=gen_mask.get_data()
        #gt_mask = np.asarray(gt_mask)
        gen_mask = np.asarray(gen_mask)
        
        print(gen_mask.shape) #64 64 40
        for camada_numero in range(0,40):
            W_gen, G_gen, CSF_gen, BET_gen = countvoxels(gen_mask[:,:,camada_numero])
            array.append([namegen+"_"+str(camada_numero),W_gen, G_gen, CSF_gen, BET_gen])
    return array

def countvoxels(numpy):
    #numpy has 3 classes to classify . BET is the sum
    grey=np.count_nonzero(numpy == [1])  #GREY
    white=np.count_nonzero(numpy == [2])  #WHITE
    csf=np.count_nonzero(numpy == [3])  #CSF
    bet=grey+white+csf
    return white,grey,csf,bet


def writeinexcel(arrayresult):
    row=0
    #                |         GROUNDTRUTH       ||     GENERATED SEG       |
    #                |       WGMCSF       |  BET ||       WGMCSF     |  BET |
    #array - 1ºNameFile, 2º W, 3º G, 4ºCSF, 5º BET, 6ºW, 7º G, 8ºCSF, 9º BET
    workbook = xlsxwriter.Workbook('/Notebooks/Marianadissertation/Rats/images/volumes.xls')
    worksheet=workbook.add_worksheet()
    for line in arrayresult:
        worksheet.write(row, 0, line[0]) #Name
        worksheet.write(row, 1, line[1]) #W      #
        worksheet.write(row, 2, line[2]) #G      # GROUND
        worksheet.write(row, 3, line[3]) #CSF    # TRUTH
        worksheet.write(row, 4, line[4]) #BET    #

        row=row+1
    workbook.close()




if __name__ == '__main__':
    arrayresult=putinfoinARRAY('/Notebooks/Marianadissertation/Rats/images/MASKC')
    writeinexcel(arrayresult)
