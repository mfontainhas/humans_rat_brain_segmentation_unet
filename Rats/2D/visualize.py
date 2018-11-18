import NumpyImage, loss_functions
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import scipy.misc
from keras import backend as K
from keras.models import load_model



def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1)


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def openModel(path):
    model = load_model(path, custom_objects={'dice_coef_loss': dice_coef_loss})
    print(model.layers[1].get_config())
    print(model.layers[1].get_weights())
    print(model.summary())
    return model


def get_featuremaps(model, layer_idx, X_batch):
    get_activations = K.function([model.layers[0].input, K.learning_phase()], [model.layers[layer_idx].output, ])
    activations = get_activations([X_batch, 0])
    return activations


def openImage(path):
    mask=nib.load("/Notebooks/Marianadissertation/Rats/images/MASK/Brain_Mask_p2b_R13_D01.nii")
    mask = mask.get_data()
    numpyt1 = nib.load(path)
    numpyt1 = NumpyImage.myNumpy(numpyt1)
    numpyt1 = numpyt1.preProcessing()
    numpyt1=np.multiply(numpyt1,mask)
    mri = np.rollaxis(numpyt1, 2).reshape(numpyt1.shape[2], numpyt1.shape[0], numpyt1.shape[1], 1)
    return mri


def visualizelayer(model, layer_num,image):
    print(np.max(image))
    print(np.min(image))
    activations = get_featuremaps(model, int(layer_num), image)
    print(np.shape(activations))
    feature_maps = activations[0][0]
    print(np.shape(feature_maps))
    num_of_featuremaps = feature_maps.shape[2]
    fig = plt.figure(figsize=(16, 16))
    plt.title("featuremaps-layer-{}".format(layer_num))
    subplot_num = int(np.ceil(np.sqrt(num_of_featuremaps)))
    for i in range(int(num_of_featuremaps)):
        ax = fig.add_subplot(subplot_num, subplot_num, i + 1)
        # ax.imshow(output_image[0,:,:,i],interpolation='nearest' ) #to see the first filter
        ax.imshow(feature_maps[:, :, i], cmap='gray')
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
    plt.show()
    fig.savefig('/Notebooks/Marianadissertation/Rats/pythonrats2D/imageswgm/featuremaps_layer-' + str(layer_num) + '.jpg')


def visualize_train_evaluate(mris_training, masks_training, mris_evaluate, masks_evaluate):
    print("shape mris_training: ", mris_training.shape)
    print("shape masks_training: ", masks_training.shape)
    print("shape mris_evaluate: ", mris_evaluate.shape)
    print("shape masks_evaluate: ", masks_evaluate.shape)

    print("numtype mris_training:", mris_training[100, :, :, 0].dtype)
    print("numtype masks_training:", masks_training[100, :, :, 0].dtype)

    print("mris training Mean value:       ", mris_training.mean())
    print("mris training Standard deviation:", mris_training.std())
    print("mris training Minimum value:    ", mris_training.min())
    print("mris training Maximum value:    ", mris_training.max())
    print("masks trainingMean value:       ", masks_training.mean())
    print("masks training Standard deviation:", masks_training.std())
    print("masks training Minimum value:    ", masks_training.min())
    print("masks training Maximum value:    ", masks_training.max())

    fig = plt.figure()
    plt.rcParams['image.cmap'] = 'gray'
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.subplot(221)
    plt.imshow(mris_training[150, :, :, 0])
    plt.subplot(222)
    plt.imshow(masks_training[150, :, :, 0])
    plt.subplot(223)
    plt.imshow(mris_training[100, :, :, 0])
    plt.subplot(224)
    plt.imshow(masks_training[100, :, :, 0])
    plt.show()
    plt.savefig('images.jpg')
    scipy.misc.imsave('Check train images/mrit.jpg', mris_training[100, :, :, 0])
    scipy.misc.imsave('Check train images/maskt.jpg', masks_training[100, :, :, 0])
    scipy.misc.imsave('Check train images/mrie.jpg', mris_evaluate[100, :, :, 0])
    scipy.misc.imsave('Check train images/maske.jpg', masks_evaluate[100, :, :, 0])

if __name__ == '__main__':
    image = openImage("/Notebooks/Marianadissertation/Rats/images/MRI/SIGMA_p2b_R13_D01_1_1_dti_b0.nii.gz")
    print(str(image.shape))
    #print(np.max(mask))
    print(np.max(image))
    print(np.min(image))
    #image=np.multiply(image,mask)
    print(str(image.shape))
    print(str(image.shape))
    image=np.reshape(image[32,:,:,:],(1,64,64,1))
    layers=[1,2,3,6,8,10,12,14,15,18,20,30,34,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50]
    model = load_model('/Notebooks/Marianadissertation/Rats/pythonrats2D/bestvalueslasttrain/Teste7 sem imagens teste (wgm/val_loss.h5',
                       custom_objects={'dice_coef_loss': loss_functions.dice_coef_loss,
                                       'dice_coef_multilabel': loss_functions.dice_coef_multilabel,
                                       'Precision': loss_functions.Precision, 'FPR': loss_functions.FPR,
                                       'FNR': loss_functions.FNR, 'specificity': loss_functions.specificity,
                                       'sensitivity': loss_functions.sensitivity})
    for i in layers:
        visualizelayer(model,i,image)