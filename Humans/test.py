import os

import NumpyImage
import loss_functions
import nibabel as nib
import numpy as np
from keras.models import load_model

img_depth, img_rows, img_cols = 40, 64, 64
# If testing wmg =True; If testing segmentation =False
WMG_SEG = True#
MODELS_DIR='/Notebooks/Marianadissertation/Humans/2D-BrainSeg/best_train_models'

def openmodel():
    trained_models = sorted(os.listdir(MODELS_DIR))
    for i in range(len(trained_models)):
        print(str(i) + " - " + trained_models[i])
    try:
        model_number = int(input('Model number: '))
    except ValueError:
        print("Not a number")

    model = load_model(MODELS_DIR+'/' + trained_models[model_number],
                       custom_objects={'Tversky_loss': loss_functions.Tversky_loss,
                                       'dice_coef_loss': loss_functions.dice_coef_loss,
                                       'Precision': loss_functions.Precision, 'FPR':loss_functions.FPR, 'FNR': loss_functions.FNR, 'specificity': loss_functions.specificity,'sensitivity': loss_functions.sensitivity})
    return model


def trymodel(pathtest, numberoffiles, model):
    files_list = sorted(os.listdir(pathtest))
    print(files_list)
    for i in range(0, len(files_list)):
        print(files_list[i])
        mri = nib.load(pathtest + files_list[i])
        affine = mri.affine
        # mri = mri.get_data()
        mri = NumpyImage.myNumpy(mri)
        mri = mri.preProcessing()  # divisao pelo max
        # preparedata
        # mri = np.concatenate(mri, 2)
        mri = np.rollaxis(mri, 2).reshape(mri.shape[2], 256, 256, 1)
        if WMG_SEG == True:
            mask = nib.load(pathtest + files_list[i + 2])
            mask = np.rollaxis(mask, 2).reshape(mask.shape[2], 256, 256, 1)
            mask = np.floor(mask)
            mask /= np.max(mask)
        print(mri.shape)
        prediction = model.predict(mri, batch_size=1)
        prediction = np.rollaxis(prediction, 0, 3).reshape(256, 256, mri.shape[0])
        # Nifti transformation
        prediction_img = nib.Nifti1Image(prediction, affine)
        generated_mask = nib.Nifti1Image(np.round(prediction), affine)
        # Saving mask and prediction
        nib.save(generated_mask,
                 '/Notebooks/Marianadissertation/Humans/2D-BrainSeg/Results/generated_' + str(i) + '.nii.gz')
        nib.save(prediction_img,
                 "/Notebooks/Marianadissertation/Humans/2D-BrainSeg/Results/prediction_" + str(i) + '.nii.gz')


if __name__ == '__main__':
    model = openmodel()
    trymodel('/Notebooks/Marianadissertation/Humans/imagestesting/', 5, model)
