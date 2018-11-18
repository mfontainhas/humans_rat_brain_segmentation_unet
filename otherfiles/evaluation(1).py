import numpy as np
import os
import nibabel as nib
from keras import utils
from sklearn.metrics import confusion_matrix
#Humans
#PATH_GENMASKS='/home/mariana/Desktop/OLDERCASES/WGMGenolder'
#PATH_GTMAKS='/home/mariana/Desktop/OLDERCASES/WGM_GT'
PATH_GTMAKS='/Notebooks/Marianadissertation/Humans/older'
PATH_GENMASKS='/Notebooks/Marianadissertation/Humans/resultsolder'
#PATH_GENMASKS='/Notebooks/Marianadissertation/Humans/2D-BrainSeg/ResultsTeens'
#PATH_GENMASKS='/Notebooks/Marianadissertation/Humans/resultsCERSC'
#PATH_GTMAKS='/Notebooks/Marianadissertation/Humans/TestImagesTeens'
#RATS
#PATH_GENMASKS='/home/mariana/Desktop/TestingRats/WGMGEN3D - 1024'
#PATH_GTMAKS='/home/mariana/Desktop/TestingRats/WGMGT'
#PATH_GENMASKS='/home/mariana/Desktop/TestingRats/genmask3d1024'
#PATH_GTMAKS='/home/mariana/Desktop/TestingRats/BETGT'
#PATH_GENMASKS='/home/mariana/Desktop/wgmsc_genteens'
#PATH_GTMAKS='/home/mariana/Desktop/wgmscteens'
sumacc=0

def evaluate_bin_one_image(ground_mask, generated_mask):
    print(ground_mask.shape)
    print(generated_mask.shape)
    #generated_mask=correct_images_size(generated_mask)
    #ground_mask=correct_images_size(ground_mask)
    TP = np.sum((np.multiply(ground_mask, generated_mask)))
    TN = np.count_nonzero((np.add(ground_mask,generated_mask)) == 0)
    FP = np.count_nonzero((np.subtract(generated_mask,ground_mask)) == 1) #gen mask1 gr 0
    FN = np.count_nonzero((np.subtract(ground_mask, generated_mask)) == 1)  # gen mask0 gr 1

    accuracy=(TP+TN)/(TP+TN+FP+FN)
    sensitivity=TP/(TP + FN)
    #precision = TP(TP+FP)
    specificity=TN/(TN+FP)
    dsc = (2 * TP) / (2 * TP + FP + FN)


    print("True Positives: "+str(TP))
    print("False Positives: " + str(FP))
    print("True Negatives: " + str(TN))
    print("False Negatives: " + str(FN))
    print("Accuracy: "+str(accuracy))
    print("Sensitivity: " + str(sensitivity))
    print("Specificity: "+ str(specificity))
    print("DSC: " + str(dsc))
    return accuracy, sensitivity, specificity,dsc

def evaluate_multiclasse_one_image(ground_mask,generated_mask,number_classes):
    print(ground_mask.shape)
    print(generated_mask.shape)
    ground_mask=utils.to_categorical(ground_mask, num_classes=number_classes)
    generated_mask=utils.to_categorical(generated_mask, num_classes=number_classes)
    print(ground_mask.shape)
    TP = 0
    TN =0
    FP=0
    FN=0
    values=[] #0 - color, 1- value dcs
    for i in range (2,number_classes):
        print("Map color: "+ str(i))
        TP = TP+np.sum((np.multiply(ground_mask[:,:,:,i], generated_mask[:,:,:,i])))
        TN = TN+np.count_nonzero((np.add(ground_mask[:,:,:,i], generated_mask[:,:,:,i])) == 0)
        FP = FP+np.count_nonzero((np.subtract(generated_mask[:,:,:,i], ground_mask[:,:,:,i])) == 1)  # gen mask1 gr 0
        FN = FN+np.count_nonzero((np.subtract(ground_mask[:,:,:,i], generated_mask[:,:,:,i])) == 1)  # gen mask0 gr 1
        print(TP)
        print(TN)
        print(FP)
        print(FN)
        accuracy = ((TP + TN) / (TP + TN + FP + FN))
        sensitivity = (TP / (TP + FN))
        # precision = TP(TP+FP)
        specificity = (TN / (TN + FP))
        dsc = (2*TP)/(2*TP + FP + FN)
        values.append([i,accuracy,sensitivity,specificity,dsc])
        print(values)
    #print("Accuracy: " + str(accuracy))
    #print("Sensitivity: " + str(sensitivity))
    #print("Specificity: " + str(specificity))
    #print("DSC: " + str(dsc))
    return values#accuracy,sensitivity, specificity,dsc


def run_all_images_wgm():
    files_list = sorted(os.listdir(PATH_GENMASKS))
    #files_list = sorted(os.listdir(PATH_GTMAKS))
    #gen_list = sorted(os.listdir(PATH_GENMASKS))
    total_values=[[0,0,0,0,0],[1,0,0,0,0],[2,0,0,0,0]]
    a=0
    print(PATH_GENMASKS)
    print(PATH_GTMAKS)
    for number in range(0,int(len(files_list))):
        #HUMANS
        if files_list[number][-11:]=="subc.nii.gz":
            a=a+1
            print(number)
            gt_mask = nib.load(PATH_GTMAKS+"/"+ files_list[number][:-22]+"_wgm.nii.gz")
            gen_mask = nib.load(PATH_GENMASKS+"/"+files_list[number])
    
    #RATS
    #    for namegen in gen_list:
    #        if (files_list[number])[9:-10] in namegen:
            #a = a + 1
#            gt_mask = nib.load(PATH_GTMAKS+"/"+files_list[number])
#            gen_mask=nib.load(PATH_GENMASKS+"/"+namegen)
            gt_mask = gt_mask.get_data()
            gen_mask=gen_mask.get_data()
            gt_mask = np.asarray(gt_mask)
            gen_mask = np.asarray(gen_mask)
            gen_mask=correct_images_size(gen_mask, 256)
            gt_mask=correct_images_size(gt_mask,256)
            values=evaluate_multiclasse_one_image(gt_mask, gen_mask,3)
            for colormap in range(len(values)):
                total_values[colormap][1]=total_values[colormap][1]+values[colormap][1]
                total_values[colormap][2]=total_values[colormap][2]+values[colormap][2]
                total_values[colormap][3]=total_values[colormap][3]+values[colormap][3]
                total_values[colormap][4]=total_values[colormap][4]+values[colormap][4]
                    
                    
                #acc,sens,spe,dsc=evaluate_multiclasse_one_image(gt_mask, gen_mask,4)
                #for i in range(values)
                #sum_acc=sum_acc+acc
                #sum_sen=sum_sen+sens
                #sum_spe=sum_spe+spe
                #sum_dsc=sum_dsc+dsc
    print(total_values)
    for colormap in range(len(total_values)):
        print(colormap)
        print("#########################################")
        print("########### Final Results ###############")
        print("#        Accuracy: "+ str(total_values[colormap][1]/a)+ "    #")
        print("#     Sensibility: " + str(total_values[colormap][2] / a) + "     #")
        print("#     Specificity: " + str(total_values[colormap][3] / a) + "    #")
        print("#             DSC: " + str(total_values[colormap][4] / a) + "     #")
        print("#########################################")

def run_all_images_bin():
    files_list = sorted(os.listdir(PATH_GENMASKS))
    sum_acc=0
    sum_sen=0
    sum_spe=0
    sum_dsc = 0
    print(files_list)
    a=0 
    for number in range(0,int(len(files_list))):
        if files_list[number][-10:]=="bet.nii.gz":
            a=a+1
            print(number)
            print(files_list[number][:-10])
            gt_mask = nib.load(PATH_GTMAKS+"/"+ files_list[number][:-21]+"_aseg.nii.gz")
            print(PATH_GTMAKS+"/"+ files_list[number][:-21]+"_aseg.nii.gz")
            gen_mask = nib.load(PATH_GENMASKS+"/"+files_list[number])
            #RATS
            #gen_mask = nib.load(PATH_GENMASKS + "/" + (files_list[file])[:-28] + "_generated3D.nii.gz")
            #HUMAN
            gt_mask = gt_mask.get_data()
            gen_mask=gen_mask.get_data()
            gt_mask = np.asarray(gt_mask)
            gen_mask = np.asarray(gen_mask)
            gen_mask = correct_images_size(gen_mask, 256)
            gt_mask = correct_images_size(gt_mask, 256)
            acc,sens,spe,dsc=evaluate_bin_one_image(gt_mask, gen_mask)
            sum_acc=sum_acc+acc
            sum_sen=sum_sen+sens
            sum_spe=sum_spe+spe
            sum_dsc = sum_dsc + dsc
    print("#########################################")
    print("########### Final Results ###############")
    print("#        Accuracy: "+ str(sum_acc/a)+ "    #")
    print("#     Sensibility: " + str(sum_sen / a) + "     #")
    print("#     Specificity: " + str(sum_spe / a) + "    #")
    print("#             DSC: " + str(sum_dsc / a) + "     #")
    print("#########################################")


def correct_images_size(numpyt1,size):
    
    if (size != numpyt1.shape[1]):
        numpyt1 = np.append(numpyt1,
                            np.zeros((numpyt1.shape[0], size - numpyt1.shape[1], numpyt1.shape[2])),
                            axis=1)
    if (size != numpyt1.shape[2]):
        numpyt1 = np.append(numpyt1,
                            np.zeros((numpyt1.shape[0], numpyt1.shape[1], size - numpyt1.shape[2])),
                            axis=2)
    if (size != numpyt1.shape[0]):
        numpyt1 = np.append(numpyt1,
                            np.zeros((size - numpyt1.shape[0], numpyt1.shape[1], numpyt1.shape[2])),
                            axis=0)
    return numpyt1

if __name__ == '__main__':
    run_all_images_wgm()