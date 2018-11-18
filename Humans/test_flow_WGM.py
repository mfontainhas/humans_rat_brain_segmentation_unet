import os

import NumpyImage , test_flow_SegBrain
import loss_functions
import nibabel as nib
import numpy as np
from keras.models import load_model
USE_BET_FILES=False
img_depth, img_rows, img_cols = 40, 64, 64
#MODELS_DIR='home/mariana/Desktop/MODELS'
MODELS_DIR='/Notebooks/Marianadissertation/Humans/2D-BrainSeg/best_train_models'
FINAL_DIR='/Notebooks/Marianadissertation/Humans/afteroldertrainNORMAL'
#FINAL_DIR='/Notebooks/Marianadissertation/Humans/resultsCERSC'
#TEST_IMAGES='/Notebooks/Marianadissertation/Humans/imagestesting/'
TEST_IMAGES='/Notebooks/Marianadissertation/Humans/TestImagesTeens/'
#TEST_IMAGES='/Notebooks/Marianadissertation/Humans/TestImagesTeens/'
BET_FILES='/Notebooks/Marianadissertation/Humans/TestImagesTeens'

def run_all_files(pathtest):
    files_list = sorted(os.listdir(pathtest))
    for i in range(73,len(files_list)):#24
        if (files_list[i])[-9:]=="T1.nii.gz":
            print(i)
            print("\n #########     Starting new image Test    ##########")
            print("Image name: " + files_list[i])
            mri = nib.load(pathtest + files_list[i])
            affine = mri.affine
            # mri = mri.get_data()
            mri = NumpyImage.myNumpy(mri)
            mri = mri.preProcessing()  # divisao pelo max
            # preparedata
            # mri = np.concatenate(mri, 2)
            mri = np.rollaxis(mri, 2).reshape(mri.shape[2], 256, 256, 1)
            if USE_BET_FILES==False:
                # 1ST - CLASSIFY IF THERE'S BRAIN OR NOT IN THE LAYER
                print("First Part - Classification of Brain and No-Brain layers")
                mri_only_brain_layers,low,high=test_flow_SegBrain.classify_brain_no_brain(mri)
                # 2ND - SEGMENT BRAIN OF THE LAYERS WITH BRAIN
                print("Second Part - Segmentation of the Brain with Only-Brain Layers")
                prediction=test_flow_SegBrain.segment_brain_no_brain(mri_only_brain_layers)
                print("      Brain Segmented, begins to set to zero all no-brain voxels")
                mask_brain = np.round(prediction)
            else:
                print("First part - Checking BET generated mask file...")
                numpymaskbinary=nib.load(BET_FILES+"/"+(files_list[i])[:-10]+"_generated_bet.nii.gz")
                numpymaskbinary = numpymaskbinary.get_data()
                mask_brain = np.floor(mri)
                mask_brain /= np.max(mask_brain)  # divisao pelo max
                print("Second part - Extract Brain...")
        
            print("   Saving BET")
            prediction= test_flow_SegBrain.reconstruct_mask(low,high,prediction)
            #4TH - Generate final files
            prediction_img, generated_mask = test_flow_SegBrain.generate_final_files(prediction,affine)
            #SAVE BET
            print(".......Saving images.........")
            nib.save(generated_mask,
                     FINAL_DIR+'/' + (files_list[i])[:-10] +'_generated_bet.nii.gz')
            
            # 3RD - SET TO ZERO ALL NO-BRAIN VOXELS
            
            mri_only_brain_voxel=np.multiply(mask_brain,mri_only_brain_layers)
            print("Third Part - Segmentation of the WGM with Only-Brain images")
            # 4TH - SEGMENT WGM OF THE IMAGE WITH ONLY BRAIN VOXELS
            prediction_wgm=segment_wgm(mri_only_brain_voxel)
            #5TD ADD Layers zero to fill the mask 256x256x256 and create final files
            print("Final Part: Reconstruction mask and generating images")
            generated_mask = generate_final_wgm(prediction_wgm,low,high,affine)
            #SAVE ALL
            print("      .......Saving images.........")
            nib.save(generated_mask,
                     FINAL_DIR+'/' + (files_list[i])[:-10] +'_generated_wgm.nii.gz')
        
    print(' ############            Finish            #############"')


def segment_wgm(mri):
   # model_segment = load_model('/Notebooks/Marianadissertation/Humans/2D-BrainSeg/best_train_models/wgm2/val_loss.h5',
   #                    custom_objects={'assimetric_loss_multilabel': loss_functions.assimetric_loss_multilabel,
   #                                    'Tversky_multilabel': loss_functions.Tversky_multilabel,
   #                                    'dice_coef_multilabel': loss_functions.dice_coef_multilabel,
   #                                    'Precision': loss_functions.Precision, 'FPR': loss_functions.FPR,
   #                                    'FNR': loss_functions.FNR,
   #                                    'sensitivity_background':loss_functions.sensitivity_background,
   #                                    'sensitivity_tissue1':loss_functions.sensitivity_tissue1,
   #                                    'sensitivity_tissue2':loss_functions.sensitivity_tissue2,
   #                                    'sensitivity_tissue3':loss_functions.sensitivity_tissue3,
   #                                    'sensitivity_tissue4':loss_functions.sensitivity_tissue4,
   #                                    'specificity': loss_functions.specificity,
   #                                    'sensitivity': loss_functions.sensitivity})
    model_segment = load_model(MODELS_DIR + '/OLD-WGM/val_loss.h5',
                               custom_objects={'Tversky_multilabel': loss_functions.Tversky_multilabel,
                                               'dice_coef_loss': loss_functions.dice_coef_loss,
                                               'dice_coef_multilabel': loss_functions.dice_coef_multilabel, 'sensitivity_background': loss_functions.sensitivity_background,
                                               'sensitivity_tissue1': loss_functions.sensitivity_tissue1,
                                               'sensitivity_tissue2': loss_functions.sensitivity_tissue2,
                                               'Precision': loss_functions.Precision, 'FPR': loss_functions.FPR,
                                               'dice_coef_multilabel_tissueMatters':loss_functions.dice_coef_multilabel_tissueMatters,
                                               'sensitivity_tissue1':loss_functions.sensitivity_tissue1,
                                               'sensitivity_tissue2':loss_functions.sensitivity_tissue2,
                                               'sensitivity_tissue3':loss_functions.sensitivity_tissue3,
                                               #'sensitivity_tissue4':loss_functions.sensitivity_tissue4,
                                               #'sensitivity_tissue5':loss_functions.sensitivity_tissue5,                                               
                                               'FNR': loss_functions.FNR, 'specificity': loss_functions.specificity,
                                               'sensitivity': loss_functions.sensitivity})
    prediction = model_segment.predict(mri, batch_size=1)
    return prediction


def generate_final_wgm(prediction,low,high,affine):
    
    #Create final images roll and argmax
    generated_mask = np.argmax(prediction, 3)  # collect info from index 3 to one dimension
    
    #ADD LAYERS PREVIOUSLY REMOVED
    print("      Shape before add other layers: " + str(generated_mask.shape))
    
    generated_mask = np.append(generated_mask,
                               np.zeros((256 - high-1, 256, 256)),
                               axis=0)
    print(high)
    print(low)
    generated_mask = np.insert(generated_mask, 0, np.zeros((low, 256, 256)), axis=0)
    
    print("      Shape after reconstructiong mask:" + str(generated_mask.shape))
    
    #Too nifti
    generated_mask = np.rollaxis(generated_mask, 0, 3).reshape(256, 256, generated_mask.shape[0])
    generated_mask = nib.Nifti1Image(generated_mask, affine)
    
    return generated_mask
    
if __name__ == '__main__':
    run_all_files(TEST_IMAGES)
