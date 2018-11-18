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
    numpyt1 = nib.load(path)
    numpyt1 = NumpyImage.myNumpy(numpyt1)
    numpyt1 = numpyt1.preProcessing()
    mri = np.rollaxis(numpyt1, 2).reshape(numpyt1.shape[2], numpyt1.shape[0], numpyt1.shape[1], 1)
    return mri


def visualizelayer(model, layer_num):
    image = openImage("/Notebooks/Marianadissertation/Humans/T1_WGM_ASEG/13_T1.nii.gz")
    mask=openImage("/Notebooks/Marianadissertation/Humans/T1_WGM_ASEG/13_aseg_sbin.nii.gz")
    image=np.multiply(image,mask)
    image=np.reshape(image[180,:,:,:],(1,256,256,1))#170
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
    fig.savefig('/Notebooks/Marianadissertation/Humans/2D-BrainSeg/imageswgm/featuremaps_layer120-' + str(layer_num) + '.jpg')


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
    #layers=[21,24,27,30,35,39,45,47,51,52]
    #layers=[22,23,25,26,28,29,32,34]
    #
    layers=[35,36,37,38,39,40,41,42,43,44,45]
    model = load_model('/Notebooks/Marianadissertation/Humans/2D-BrainSeg/best_train_models/OLD-WGM/val_loss.h5',
                       custom_objects={'assimetric_loss_multilabel': loss_functions.assimetric_loss_multilabel,
                                       'Tversky_multilabel': loss_functions.Tversky_multilabel,
                                       'dice_coef_multilabel': loss_functions.dice_coef_multilabel,
                                       'Precision': loss_functions.Precision, 'FPR': loss_functions.FPR,
                                       'FNR': loss_functions.FNR,
                                       'sensitivity_background':loss_functions.sensitivity_background,
                                       'sensitivity_tissue1':loss_functions.sensitivity_tissue1,
                                       'sensitivity_tissue2':loss_functions.sensitivity_tissue2,
                                       'sensitivity_tissue3':loss_functions.sensitivity_tissue3,
                                       'specificity': loss_functions.specificity,
                                       'sensitivity': loss_functions.sensitivity})
    for i in layers:
        visualizelayer(model,i)