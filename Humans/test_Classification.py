import os

import NumpyImage
import loss_functions
import nibabel as nib
import numpy as np
from keras.models import load_model

img_depth, img_rows, img_cols = 40, 64, 64
# If testing wmg =True; If testing segmentation =False
WMG_SEG = True
MODELS_DIR='/Notebooks/Marianadissertation/Humans/2D-BrainSeg/best_train_models'


def openmodel():
    trained_models = sorted(os.listdir(MODELS_DIR))
    for i in range(len(trained_models)):
        print(str(i) + " - " + trained_models[i])
    try:
        model_number = int(input('Model number: '))
    except ValueError:
        print("Not a number")

    model = load_model(MODELS_DIR +'/'+ trained_models[model_number],
                       custom_objects={'Tversky_loss': loss_functions.Tversky_loss,
                                       'dice_coef_loss': loss_functions.dice_coef_loss,
                                       'Precision': loss_functions.Precision, 'FPR': loss_functions.FPR,
                                       'FNR': loss_functions.FNR, 'specificity': loss_functions.specificity,
                                       'sensitivity': loss_functions.sensitivity})
    return model


def trymodel(pathtest, numberoffiles, model):
    files_list = sorted(os.listdir(pathtest))
    for i in range(0, len(files_list)):
        slices_brain = []
        print(files_list[i])
        mri = nib.load(pathtest + files_list[i])
        affine = mri.affine
        # mri = mri.get_data()
        mri = NumpyImage.myNumpy(mri)
        mri = mri.preProcessing()  # divisao pelo max
        # preparedata
        # mri = np.concatenate(mri, 2)
        mri = np.rollaxis(mri, 2).reshape(mri.shape[2], 256, 256, 1)
        print(mri.shape)
        prediction = model.predict(mri, batch_size=1)
        for slice_number in range(0,len(prediction)):
            if prediction[slice_number] >0.5:
                slices_brain.append(slice_number)
        print(slices_brain)
        low_layer, high_layer=clean_list(slices_brain)
        print(low_layer)
        print(high_layer)
        #all_mris_split=np.split(mri, [low_layer, high_layer])
        #mri_only_brain=all_mris_split[1]
        #generated_mri_only_brain = nib.Nifti1Image(mri_only_brain, affine)
        # Saving mask and prediction

        #nib.save(generated_mri_only_brain,
                 #'/Notebooks/Marianadissertation/Humans/2D-BrainSeg/Results/mri_only_brain_' + str(i) + '.nii.gz')


def clean_list(array):
    slice_alone=[]
    for i in range(2, len(array) - 2):
        if array[i + 1]!=array[i]+1 and array[i + 2]!=array[i]+2 and array[i - 1]!=array[i]-1 and array[i - 2]!=array[i]-2:
            try:
                slice_alone.append(i)
            except:
                print("Erro in the first 2 layers and last two")
    new_array=np.delete(array, slice_alone, axis=0)
    #returns the  lowest and highest layer of the x axis
    return new_array[0], new_array[len(new_array)-1]


if __name__ == '__main__':
    model = openmodel()
    trymodel('/Notebooks/Marianadissertation/Humans/imagestesting/', 5, model)
