import os

import NumpyImage
import loss_functions
import nibabel as nib
import numpy as np
from keras.models import load_model

img_depth, img_rows, img_cols = 40, 64, 64

#LOCAL COMPUTER
#MODELS_DIR='/home/mariana/Desktop/MODELS'
#FINAL_DIR='/home/mariana/Desktop/generated'
#DIR_TEST_IMAGES='/home/mariana/Desktop/MODELS/images/'

#MIVBOX
MODELS_DIR='/Notebooks/Marianadissertation/Humans/2D-BrainSeg/best_train_models'
FINAL_DIR='/Notebooks/Marianadissertation/Humans/resultsmp2rage/'
DIR_TEST_IMAGES='/Notebooks/Marianadissertation/Humans/testg/'

def run_all_files(pathtest):
    files_list = sorted(os.listdir(pathtest))
    for i in range(0,len(files_list)):
        if (files_list[i])[-9:]=="T1.nii.gz":
            print("Image name: " +files_list[i])
            mri = nib.load(pathtest + files_list[i])
            affine = mri.affine
            # mri = mri.get_data()
            mri = NumpyImage.myNumpy(mri)
            mri = mri.preProcessing()  # divisao pelo max
            # preparedata
            # mri = np.concatenate(mri, 2)
            mri = np.rollaxis(mri, 2).reshape(mri.shape[2], 256, 256, 1)
            print(mri.shape)
            # 1ST - CLASSIFY IF THERE'S BRAIN OR NOT IN THE LAYER
            print("First Part - Classification of Brain and No-Brain layers")
            mri_only_brain,low,high=classify_brain_no_brain(mri)
            # 2ND - SEGMENT BRAIN OF THE LAYERS WITH BRAIN
            print("Second Part - Segmentation of the Brain")
            prediction=segment_brain_no_brain(mri)#_only_brain)
            #3RD ADD Layers zero to fill the mask 256x256x256
            print("Final Part: Reconstruction mask and generating images")
            print(str(prediction.shape))
            prediction= reconstruct_mask(low,high,prediction)
            #4TH - Generate final files
            prediction_img, generated_mask = generate_final_files(prediction,affine)
            #SAVE ALL
            print(".......Saving images.........")
            nib.save(generated_mask,
                     FINAL_DIR+'/'+(files_list[i])[:-10]+'_generated_bet.nii.gz')
            nib.save(prediction_img,
                     FINAL_DIR+"/"+(files_list[i])[:-10]+'_predict_bet.nii.gz')
    print('#####  FINISH ####')

def classify_brain_no_brain(mri):
    mri_only_brain = []
    slices_brain = []
    model_classify = load_model(MODELS_DIR + '/C4/val_loss.h5',
                                custom_objects={'Tversky_loss': loss_functions.Tversky_loss,
                                                'dice_coef_loss': loss_functions.dice_coef_loss,
                                                'Precision': loss_functions.Precision, 'FPR': loss_functions.FPR,
                                                'FNR': loss_functions.FNR, 'specificity': loss_functions.specificity,
                                                'sensitivity': loss_functions.sensitivity})
    
    prediction = model_classify.predict(mri, batch_size=1)
    print(prediction)
    for slice_number in range(0, len(prediction)):
        if prediction[slice_number] > 0.5:
            slices_brain.append(slice_number)
    print(slices_brain)
    low, high = clean_list(slices_brain)
    for i in range(low,high+1):
        #print(i)
        mri_only_brain.append(mri[i])
    mri_only_brain = np.asarray(mri_only_brain)
    return mri_only_brain,low,high

def clean_list(array):
    slice_alone=[]
    
    for i in range(0, len(array)):   
        if i<=2 and array[i + 1]!=array[i]+1 and array[i + 2]!=array[i]+2:
                slice_alone.append(i)
        elif i>2 and i<len(array)-2 and array[i + 1]!=array[i]+1 and array[i + 2]!=array[i]+2 and array[i - 1]!=array[i]-1 and array[i - 2]!=array[i]-2:
            slice_alone.append(i)
        elif i>=len(array)-2 and array[i - 1]!=array[i]-1 and array[i - 2]!=array[i]-2:
            slice_alone.append(i)
        
    #print(array)
    new_array=np.delete(array, slice_alone, axis=0)
    #returns list of array
    print(new_array)
    print("      Low value starting brain: " + str(new_array[0]))
    print("      High value ending brain: " + str(new_array[len(new_array)-1]))
    return new_array[0],new_array[len(new_array)-1]

def segment_brain_no_brain(mri):
    model_segment = load_model(MODELS_DIR + '/OLD-BET/val_loss.h5', #'/H2/val_acc.h5'
                                custom_objects={'Tversky_loss': loss_functions.Tversky_loss,
                                                'dice_coef_loss': loss_functions.dice_coef_loss,
                                                'Precision': loss_functions.Precision, 'FPR': loss_functions.FPR,
                                                'FNR': loss_functions.FNR, 'specificity': loss_functions.specificity,
                                                'sensitivity': loss_functions.sensitivity})
    prediction = model_segment.predict(mri, batch_size=1)
    return prediction

def reconstruct_mask(low,high,prediction):
    #ADD LAYERS PREVIOUSLY REMOVED
    print("      Shape before add other layers: " + str(prediction.shape))
    prediction = np.append(prediction,
                               np.zeros((256 - high-1, 256, 256,1)),
                               axis=0)
    prediction = np.insert(prediction, 0, np.zeros((low, 256, 256,1)), axis=0)
    print("      Shape after reconstructiong mask:" + str(prediction.shape))
    return prediction

def generate_final_files(prediction,affine):
    
    prediction = np.rollaxis(prediction, 0, 3).reshape(256, 256, prediction.shape[0])
    # Nifti transformation
    prediction_img = nib.Nifti1Image(prediction, affine)
    generated_mask = nib.Nifti1Image(np.round(prediction), affine)
    return prediction_img, generated_mask
    
if __name__ == '__main__':
    run_all_files(DIR_TEST_IMAGES)
