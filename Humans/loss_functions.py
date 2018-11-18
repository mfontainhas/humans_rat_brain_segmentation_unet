from keras import backend as K
import numpy as np

# %% LOSS
smooth = 1.
smooth2 = 1e-5


#BINARY METRICS
def FP(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    return np.sum(y_pred_f * (1 - y_true_f))


def FN(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    return np.sum((1 - y_pred_f) * y_true_f)


def TP(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    return np.sum(y_pred_f * y_true_f)


def TN(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    return np.sum((1 - y_pred_f) * (1 - y_true_f))


def FPR(y_true, y_pred):
    # fallout
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    return 1.0 * (K.sum(y_pred_f * (1 - y_true_f))) / (K.sum(1 - y_true_f) + smooth2)


def FNR(y_true, y_pred):
    # miss rate
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    return 1.0 * (K.sum((1 - y_pred_f) * y_true_f)) / (K.sum(y_true_f) + smooth2)


def sensitivity(y_true, y_pred):
    # TPR, recall, hit rate
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    return 1.0 * (K.sum(y_pred_f * y_true_f)) / (K.sum(y_true_f) + smooth2)


def specificity(y_true, y_pred):
    # TNR
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    return 1.0 * (K.sum((1 - y_pred_f) * (1 - y_true_f))) / (K.sum(1 - y_true_f) + smooth2)


# return np.sum( (1.-y_pred_f)*y_true_f  )/(1.0*np.sum( y_true_f )+smooth2)
# return 1-(np.sum( y_pred_f*y_true_f )+smooth2)/(1.0*np.sum(y_true_f)+smooth2 ) # 1 - TPR

def Precision(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    return (1.0 * K.sum(y_pred_f * y_true_f) + smooth2) / (
                K.sum(y_pred_f * y_true_f) + K.sum(y_pred_f * (1.0 - y_true_f)) + smooth2)


def Tversky(y_true, y_pred, alpha=0.3, beta=0.7): #alpha low for little 1 and lot 0
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    G_P = alpha * K.sum((1 - y_true_f) * y_pred_f)  # G not P
    P_G = beta * K.sum(y_true_f * (1 - y_pred_f))  # P not G
    return (intersection + smooth) / (intersection + smooth + G_P + P_G)
    
def Tversky_loss(y_true, y_pred):
    return 1-Tversky(y_true, y_pred)

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

def assimetric_loss(y_true, y_pred,beta=1.5):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    G_P =  K.sum((1 - y_pred_f) * y_true_f)
    P_G =  K.sum(y_pred_f * (1 - y_true_f))  # P not G
    beta_2=beta*beta
    return ((1+beta_2)*(intersection) / ((1+beta_2)*intersection +  beta_2*G_P + beta_2* P_G))


#MULTILABEL METRICS
'''
def dice_coef_multilabel(y_true, y_pred, numLabels=2):
    dice=0
    for index in range(0,numLabels):
        dice -= (dice_coef(y_true[:,:,:,index], y_pred[:,:,:,index]))  
    return dice
    '''

def dice_coef_multilabel(y_true, y_pred, numLabels=2):
    dice=0
    dice -= (dice_coef(y_true[:,:,:,0], y_pred[:,:,:,0]))*0.2
    dice -= (dice_coef(y_true[:,:,:,1], y_pred[:,:,:,1])) *0.4
    dice -= (dice_coef(y_true[:,:,:,2], y_pred[:,:,:,2]))  *0.4
    #for index in range(0,numLabels):
    #    dice -= (dice_coef(y_true[:,:,:,index], y_pred[:,:,:,index]))  
    return dice

def assimetric_loss_multilabel(y_true, y_pred,numLabels=2): #WGM
    ass=0 
    for index in range(0,numLabels):
        ass -= assimetric_loss(y_true[:,:,:,index], y_pred[:,:,:,index])
    return ass

def assimetric_loss_multilabel_CS(y_true, y_pred,numLabels=3): #CS
    ass=0
    ass -= assimetric_loss(y_true[:,:,:,0], y_pred[:,:,:,0])*0.03
    for index in range(1,numLabels-2):
        ass -= assimetric_loss(y_true[:,:,:,index], y_pred[:,:,:,index])*0.08
    ass -= assimetric_loss(y_true[:,:,:,3], y_pred[:,:,:,3])*0.35
    ass -= assimetric_loss(y_true[:,:,:,4], y_pred[:,:,:,4])*0.55
    return ass


def Tversky_multilabel(y_true, y_pred, numLabels=2):
    dice=0
    for index in range(0,numLabels): #Não me interessa o fun
        dice -= Tversky(y_true[:,:,:,index], y_pred[:,:,:,index])
    return dice

def dice_coef_multilabel_tissueMatters(y_true, y_pred, numLabels=2):
    dice_tissues=0
    dice_background = dice_coef(y_true[:, :, :, 0], y_pred[:, :, :, 0])
    for index in range(1,numLabels):
        dice_tissues -= dice_coef(y_true[:,:,:,index], y_pred[:,:,:,index])
    dice=0.7*dice_tissues-0.3*dice_background
    return dice

def sensitivity_background(y_true, y_pred,index=0):
    # TPR, recall, hit rate
    y_true_f = K.flatten(y_true[:,:,:,index])
    y_pred_f = K.flatten(y_pred[:,:,:,index])
    return 1.0 * (K.sum(y_pred_f * y_true_f)) / (K.sum(y_true_f) + smooth2)

def accuracy_tissue1(y_true, y_pred):
    return K.cast(K.equal(K.argmax(y_true, axis=-1)[1],
                          K.argmax(y_pred, axis=-1)[1]),
                  K.floatx())

def accuracy_tissue2(y_true, y_pred):
    return K.cast(K.equal(K.argmax(y_true, axis=-1)[2],
                          K.argmax(y_pred, axis=-1)[2]),
                  K.floatx())

def sensitivity_tissue1(y_true, y_pred,index=1):
    # TPR, recall, hit rate
    y_true_f = K.flatten(y_true[:,:,:,index])
    y_pred_f = K.flatten(y_pred[:,:,:,index])
    return 1.0 * (K.sum(y_pred_f * y_true_f)) / (K.sum(y_true_f) + smooth2)

def sensitivity_tissue2(y_true, y_pred,index=2):
    # TPR, recall, hit rate
    y_true_f = K.flatten(y_true[:,:,:,index])
    y_pred_f = K.flatten(y_pred[:,:,:,index])
    return 1.0 * (K.sum(y_pred_f * y_true_f)) / (K.sum(y_true_f) + smooth2)

def sensitivity_tissue3(y_true, y_pred,index=3):
    # TPR, recall, hit rate
    y_true_f = K.flatten(y_true[:,:,:,index])
    y_pred_f = K.flatten(y_pred[:,:,:,index])
    return 1.0 * (K.sum(y_pred_f * y_true_f)) / (K.sum(y_true_f) + smooth2)
'''
def sensitivity_tissue4(y_true, y_pred,index=4):
    # TPR, recall, hit rate
    y_true_f = K.flatten(y_true[:,:,:,index])
    y_pred_f = K.flatten(y_pred[:,:,:,index])
    return 1.0 * (K.sum(y_pred_f * y_true_f)) / (K.sum(y_true_f) + smooth2)

def sensitivity_tissue5(y_true, y_pred,index=5):
    # TPR, recall, hit rate
    y_true_f = K.flatten(y_true[:,:,:,index])
    y_pred_f = K.flatten(y_pred[:,:,:,index])
    return 1.0 * (K.sum(y_pred_f * y_true_f)) / (K.sum(y_true_f) + smooth2)
    '''