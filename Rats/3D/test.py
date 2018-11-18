import nibabel as nib
import numpy as np
import os
import loss_functions
from keras.models import load_model

img_depth, img_rows, img_cols = 40, 64, 64
WMG_SEG=True #True-Segmentation more than 1 tissue;  False-Segmentation of 1 and 0
MODELS_DIR='/Notebooks/Marianadissertation/Rats/pythonrats3D/bestvalueslasttrain'
FINAL_DIR='/Notebooks/Marianadissertation/Rats/pythonrats3D/RESULTS TEENS'
IMAGE_TESTING_DIR='/Notebooks/Marianadissertation/Rats/images/MRI'
MASK_TESTING_DIR='/Notebooks/Marianadissertation/Rats/images/MASKC'

def segment_brain(mri):

    model = load_model(MODELS_DIR+'/Teste5seg/val_loss.h5', custom_objects={'Tversky_loss': loss_functions.Tversky_loss,
                                       'dice_coef_loss': loss_functions.dice_coef_loss,
                                       'Precision': loss_functions.Precision, 'FPR': loss_functions.FPR,
                                       'FNR': loss_functions.FNR, 'specificity': loss_functions.specificity,
                                       'sensitivity': loss_functions.sensitivity})
    prediction = model.predict(mri, batch_size=1)
    return prediction

def segment_wgm(mri):

    model = load_model(MODELS_DIR+'/WGM.final/val_loss.h5', custom_objects={'Tversky_loss': loss_functions.Tversky_loss,
                                       'dice_coef_multilabel': loss_functions.dice_coef_multilabel, 'dice_coef_loss': loss_functions.dice_coef_loss,
                                       'Precision': loss_functions.Precision, 'FPR': loss_functions.FPR,
                                       'FNR': loss_functions.FNR, 'specificity': loss_functions.specificity,
                                       'sensitivity': loss_functions.sensitivity})
    prediction = model.predict(mri, batch_size=1)
    return prediction

def trymodel(path_testingimages):
    files_list = sorted(os.listdir(path_testingimages))
    for i in range(144,len(files_list)):
        a=[]
        splits=files_list[i].split("_")
        name=splits[1]+"_"+splits[2]+"_"+splits[3]
        print(files_list[i])
        mri = nib.load(path_testingimages + '/'+files_list[i])
        affine = mri.affine
        mri = mri.get_data()
        mri = np.floor(mri)
        mri /= np.max(mri)
        
        # 1st segment brain with original mask
        mri = np.reshape(mri,[1, 64, 64,40,1 ])           
        print("First Part - Segmentation of the Brain")
        prediction = segment_brain(mri)
        if WMG_SEG==False:
            # Nifti transformation
            prediction = np.reshape(prediction,[64, 64, 40])
            prediction_img = nib.Nifti1Image(prediction, affine)
            generated_mask=nib.Nifti1Image(np.round(prediction),affine)
            # Saving mask and prediction
            nib.save(generated_mask, FINAL_DIR+
                     '/' + str(files_list[i][:-7]) + '_generated3D.nii.gz')
            nib.save(prediction_img, FINAL_DIR+'/' + str(files_list[i][:-7]) + '_pred3D.nii.gz')
        else:
            prediction=np.multiply(np.round(prediction),mri)
            prediction = np.reshape(prediction,[64, 64, 40])
            prediction_img = nib.Nifti1Image(prediction, affine)
            nib.save(prediction_img, FINAL_DIR+
                     '/' + str(files_list[i][:-7]) + '_wgm3D.nii.gz')

def segment_multiple_masks():
    files_list_MASKC = sorted(os.listdir(MASK_TESTING_DIR))
    numero=0
    for i in range(0, len(files_list_MASKC)):
        splits_MASKC=files_list_MASKC[i].split("_")
        nameMASKC=splits_MASKC[1]+"_"+splits_MASKC[2]+"_"+splits_MASKC[3]
        mri_list=sorted(os.listdir(IMAGE_TESTING_DIR))
        for MRI_NAME in mri_list:
            if nameMASKC in MRI_NAME:
                print(nameMASKC)
                print(MRI_NAME)
                numero=1+numero
                print(numero)
                if numero>116: #136
                    pathMRI=IMAGE_TESTING_DIR+"/"+MRI_NAME
                    print(numero)
                    #print(nameMASKC)
                    mri = nib.load(pathMRI)
                    affine = mri.affine
                    mri = mri.get_data()
                    mri = np.floor(mri)
                    mri /= np.max(mri)
                    # preparedata
                    print(mri.shape)
                    mri = np.reshape(mri,[1, 64, 64,40,1 ])    
                    print(mri.shape)
                    print("First Part - Segmentation of the Brain")
                    prediction = segment_brain(mri)
                    print("Second Part - Predict white matter and gray")
                    mri_only_brain=np.multiply(mri,np.round(prediction))
                    prediction=segment_wgm(mri_only_brain)                    
                    generated_mask=np.argmax(prediction,4)
                    generated_mask = np.reshape( generated_mask,[64, 64, 40])
                    #generated_mask = np.round(prediction)
                    generated_mask=nib.Nifti1Image(generated_mask,affine)
                    nib.save(generated_mask,FINAL_DIR+
                             '/' + str(MRI_NAME[:-7]) + '_wgm3D.nii.gz')
    print("FINISH")


def predictfunction(prediction , oneclassification): #For diferent classification = False
    if oneclassification==True:
        print("Predict one tissue")
        prediction = np.reshape(prediction,[64, 64, 40])
        generated_mask = np.round(prediction) #If>0.5 =1 if<0.5=0
    else:
        print("Predict multiple tissues")
        print(prediction.shape)
        generated_mask=np.argmax(prediction,4) #collect info from index 3 to one dimension
        print(generated_mask.shape)
        generated_mask = np.reshape(generated_mask,[64, 64, 40])
        print(generated_mask.shape)
    return generated_mask
                          
if __name__ == '__main__':
    segment_multiple_masks()
    #trymodel(IMAGE_TESTING_DIR)
