from keras import backend as K
import numpy as np
#%% LOSS
smooth = 1.
smooth2 = 1e-5 

#def confusion(TRUE_LIST,PRED_LIST):
#    for i, (y_true,y_pred) in enumerate(zip(TRUE_LIST,PRED_LIST )):
def FP(y_true,y_pred):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred) 
        return np.sum( y_pred_f*(1-y_true_f) )
    
def FN(y_true,y_pred):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred) 
        return np.sum( (1-y_pred_f)*y_true_f )
    
def TP(y_true,y_pred):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred) 
        return np.sum( y_pred_f*y_true_f )
    
def TN(y_true,y_pred):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred) 
        return np.sum( (1-y_pred_f)*(1-y_true_f) )

    
def FPR(y_true,y_pred):
    #fallout
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)        
        return 1.0*(K.sum( y_pred_f*(1-y_true_f) ) )/(K.sum(1-y_true_f)+smooth2)

def FNR(y_true,y_pred):
    #miss rate
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        return 1.0*( K.sum( (1-y_pred_f)*y_true_f ) )/ ( K.sum( y_true_f ) +smooth2 )  


def sensitivity(y_true,y_pred):
    #TPR, recall, hit rate
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        return 1.0*( K.sum( y_pred_f*y_true_f ) )/ ( K.sum( y_true_f ) +smooth2 )  

def specificity(y_true,y_pred):
    #TNR
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        return 1.0*( K.sum( (1-y_pred_f)*(1-y_true_f) ) )/ ( K.sum( 1-y_true_f ) +smooth2 )  
    
        #return np.sum( (1.-y_pred_f)*y_true_f  )/(1.0*np.sum( y_true_f )+smooth2)
        #return 1-(np.sum( y_pred_f*y_true_f )+smooth2)/(1.0*np.sum(y_true_f)+smooth2 ) # 1 - TPR
        
def Precision(y_true,y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)  
    return (1.0*K.sum( y_pred_f*y_true_f ) +smooth2)/(K.sum( y_pred_f*y_true_f ) + K.sum( y_pred_f*(1.0-y_true_f) )+smooth2)


def Tversky(y_true, y_pred, alpha=0.3,beta=0.7):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    G_P = alpha*K.sum( (1-y_true_f) * y_pred_f  ) # G not P
    P_G = beta*K.sum(  y_true_f * (1-y_pred_f) )  # P not G
    return (intersection + smooth )/(intersection + smooth +G_P +P_G )
    
def Tversky_loss(y_true, y_pred):
    return 1-Tversky(y_true, y_pred)

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

def dice_coef_multilabel(y_true, y_pred, numLabels=4):
    dice=0
    for index in range(numLabels):
        dice -= dice_coef(y_true[:,:,:,:,index], y_pred[:,:,:,:,index])
    return dice


