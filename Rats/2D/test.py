import os

import NumpyImage
import loss_functions
import nibabel as nib
import numpy as np
from keras.models import load_model

img_depth, img_rows, img_cols = 40, 64, 64
# If testing wmg =True; If testing segmentation =False
WMG_SEG = False#
MODELS_DIR='/Notebooks/Marianadissertation/Rats/pythonrats2D/bestvalueslasttrain'
FINAL_DIR='/Notebooks/Marianadissertation/Rats/pythonrats2D/RESULTS TEENS'
IMAGE_TESTING_DIR='/Notebooks/Marianadissertation/Rats/images/MRI'
MASK_TESTING_DIR='/Notebooks/Marianadissertation/Rats/images/MASKC'
def segment_brain(mri):

    model = load_model(MODELS_DIR+'/S5(sem imagens teste) 0.9903 acc/val_loss.h5', custom_objects={'Tversky_loss': loss_functions.Tversky_loss,
                                       'dice_coef_loss': loss_functions.dice_coef_loss,
                                       'Precision': loss_functions.Precision, 'FPR': loss_functions.FPR,
                                       'FNR': loss_functions.FNR, 'specificity': loss_functions.specificity,
                                       'sensitivity': loss_functions.sensitivity})
    prediction = model.predict(mri, batch_size=1)
    return prediction

def segment_wgm(mri):

    model = load_model(MODELS_DIR+'/WGM s_imagens_test/val_loss.h5', custom_objects={'Tversky_loss': loss_functions.Tversky_loss,
                                       'dice_coef_multilabel': loss_functions.dice_coef_multilabel, 'dice_coef_loss': loss_functions.dice_coef_loss,
                                       'Precision': loss_functions.Precision, 'FPR': loss_functions.FPR,
                                       'FNR': loss_functions.FNR, 'specificity': loss_functions.specificity,
                                       'sensitivity': loss_functions.sensitivity})
    prediction = model.predict(mri, batch_size=1)
    return prediction

def segment_multiple_brains(path_testingimages):
    files_list = sorted(os.listdir(path_testingimages))
    
    for i in range(144, len(files_list)):
        print(files_list[i])
        mri = nib.load(path_testingimages + '/'+files_list[i])
        affine = mri.affine
        mri = mri.get_data()
        mri = np.floor(mri)
        mri /= np.max(mri)
        # preparedata
        print(mri.shape)
        mri = np.rollaxis(mri, 2).reshape(mri.shape[2], 64, 64, 1)            
        print(mri.shape)
        print("First Part - Segmentation of the Brain")
        prediction = segment_brain(mri)
        if WMG_SEG==False: 
            print("Saving Brain Segmentation Mask")
            # Nifti transformation
            prediction_img = nib.Nifti1Image(prediction, affine)
            prediction = np.rollaxis(prediction, 0, 3).reshape(64, 64, mri.shape[0])
            generated_mask = nib.Nifti1Image(np.round(prediction), affine)
            
            # Saving mask and prediction
            nib.save(generated_mask,FINAL_DIR+
                     '/' + str(files_list[i][:-7]) + '_generated.nii.gz')
            nib.save(prediction_img,FINAL_DIR+'/' + str(files_list[i][:-7]) + '_pred.nii.gz')
        else:
            print("Second Part - Predict white matter and gray")
            mri_only_brain=np.multiply(mri,np.round(prediction))
            prediction=segment_wgm(mri_only_brain)
            generated_mask=np.argmax(prediction,3)
            generated_mask = np.rollaxis(generated_mask, 0, 3).reshape(64, 64, mri.shape[0])
            generated_mask=nib.Nifti1Image(generated_mask,affine)
            nib.save(generated_mask,FINAL_DIR+
                     '/' + str(files_list[i][:-7]) + '_genwgm.nii.gz')

            
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
                if numero>110:
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
                    mri = np.rollaxis(mri, 2).reshape(mri.shape[2], 64, 64, 1)            
                    print(mri.shape)
                    print("First Part - Segmentation of the Brain")
                    prediction = segment_brain(mri)
                    print("Second Part - Predict white matter and gray")
                    mri_only_brain=np.multiply(mri,np.round(prediction))
                    prediction=segment_wgm(mri_only_brain)
                    generated_mask=np.argmax(prediction,3)
                    generated_mask = np.rollaxis(generated_mask, 0, 3).reshape(64, 64, mri.shape[0])
                    generated_mask=nib.Nifti1Image(generated_mask,affine)
                    nib.save(generated_mask,FINAL_DIR+
                             '/' + str(MRI_NAME[:-7]) + '_genwgm.nii.gz')



if __name__ == '__main__':
    #segment_multiple_brains(IMAGE_TESTING_DIR)
    segment_multiple_masks()

