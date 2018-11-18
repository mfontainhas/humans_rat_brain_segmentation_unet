import Object_model_2DUNET,Object_model_2DUNET_Class, loss_functions, writegraphs
import Managefiles2D, ManageFilesMulti
from keras import backend as K
import tensorflow as tf
import time
import sys
import psutil
from keras.callbacks import  ModelCheckpoint , EarlyStopping
import keras
import os, importlib
from keras import losses, optimizers
from random import shuffle
#from keras.models import load_weights

##################  BEGIN VARIABLES  #################################
#Prepare files info
FILES_PREP=True #True-Prepare files from nii    ;   False - Open already saved numpys
N_FILES=160
IMAGE_MRI_PATH='/Notebooks/Marianadissertation/Rats/images/MRI'
IMAGE_MASK_BIN_PATH='/Notebooks/Marianadissertation/Rats/images/MASK'
IMAGE_MASK_PATH='/Notebooks/Marianadissertation/Rats/images/MASKC' #"MASK" for Brain Seg or "MASKC" for Brain Classification
MODEL_FINAL_SAVE='/Notebooks/Marianadissertation/Rats/pythonrats2D/final_'
MODEL_SAVE_BEST='/Notebooks/Marianadissertation/Rats/pythonrats2D/bestvalueslasttrain'
SPLIT_PERCENTAGE = 1 #Define percentage of Training images, the rest will be evaluation
#Training info
OPTIMIZER=1  #0- adam; 1- sgd
learning_rate=0.0005  # lr=learning_rate, decay=decay_rate
decay_rate=0.00005
EPOCHS = 1000
SPLIT_PERCENTAGE = 1 #Define percentage of Training images, the rest will be evaluation
BATCH_SIZE=5
VERBOSE_TRAIN = 1 #0-only text, 1-generating images
VERBOSE_imageGenerator = 1 #0-no train comments, 1-text each epoch
PRINT_NUMBER = 20 #Choose a number to print in a multiple epoch
SELECTED_LAYERS=[3,6,10,20,30,35,37] #Select only some Layers to print featuremaps
#################  ENDING VARIABLES  #################################

def definetrainfunctions():
	if OPTIMIZER==0:
		function_optmizer=optimizers.Adadelta()#lr=learning_rate)
	elif OPTIMIZER==1:
		function_optmizer=optimizers.SGD()
	print("Optimizer: "+str(function_optmizer))
	return function_optmizer

def definetrainevaluate(mris, masks):  # split train data and evaluate data
    split_len = int(len(mris) * SPLIT_PERCENTAGE)
    mris_training = mris[:split_len]
    masks_training = masks[:split_len]
    mris_evaluate = mris[split_len:]
    masks_evaluate = masks[split_len:]
    return mris_training, masks_training, mris_evaluate, masks_evaluate

def compiletrain(mris_training, masks_training, model,function_optmizer):
    print("\n ########### TRAINING ############\n")
    #Choose in optmizer sgd, load weights
    all_metrics=['accuracy',loss_functions.dice_coef_loss,loss_functions.Precision,loss_functions.FPR,loss_functions.FNR,loss_functions.specificity,loss_functions.sensitivity]
    model.compile(loss=loss_functions.dice_coef_multilabel, optimizer='sgd', metrics=all_metrics)
    #Save graphs
    tensorboard=writegraphs.TrainValTensorBoard(write_graph=False)
    #Save best val_acc ,val_loss,val_pre
    m1=ModelCheckpoint(MODEL_SAVE_BEST+"/val_loss.h5", monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    m2=ModelCheckpoint(MODEL_SAVE_BEST+"/val_acc.h5", monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    m3=ModelCheckpoint(MODEL_SAVE_BEST+"/bestweights_vall_acc.hdf5", monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', period=1)
    m4=ModelCheckpoint(MODEL_SAVE_BEST+"/bestweights_vall_loss.hdf5", monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', period=1)
    #Early Stoppping
    e1=EarlyStopping(monitor='val_loss', min_delta=0, patience=250, verbose=0, mode='auto')
    e2=EarlyStopping(monitor='val_acc', min_delta=0, patience=250, verbose=0, mode='auto')
    #Fit model with images, begin training
    history = model.fit(mris_training, masks_training, batch_size=BATCH_SIZE, epochs=EPOCHS , verbose=1, validation_split=0.2,initial_epoch=0, callbacks=[tensorboard,m1,m2,m3,m4,e1,e2])
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    model.save("/Notebooks/Marianadissertation/Rats/pythonrats2D/logs/model/" + timestamp + ".h5")
    return model, history

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

def evaluate(mris_evaluate,masks_evaluate,model,history):
    p = psutil.Process()
    cpu_time = p.cpu_times()[0]
    mem = p.memory_info()[0]
    print("###########Evaluation############\n")
    scores = model.evaluate(mris_evaluate, masks_evaluate)
    print("\n Metric: %s: %.2f%%\n" % (model.metrics_names[1], scores[1]*100))
    with open("/Notebooks/Marianadissertation/Rats/pythonrats2/logs/model/log.csv", "a") as myfile:
        myfile.write(",")
        myfile.write("\"" + ascii(model).strip() + "\",")
        myfile.write("{},".format(history.history['loss'][-1]))
        myfile.write("{},".format(history.history['acc'][-1]))
        myfile.write("{},".format(history.history['val_acc'][-1]))
        myfile.write("{},".format(history.history['val_loss'][-1]))
        myfile.write("{},".format(history.history['val_Precision'][-1]))
        myfile.write("{},".format(mem))
        myfile.write("{},\n".format(cpu_time))
        
def set_keras_backend(backend):
    print("A acertar o backend e libertar memória da grafica")
    if K.backend() != backend:
        os.environ['KERAS_BACKEND'] = backend
        importlib.reload(K)
        assert K.backend() == backend
    if backend == "tensorflow":
        K.get_session().close()
        cfg = K.tf.ConfigProto()
        cfg.gpu_options.allow_growth = True
        #cfg.gpu_options.allocator_type = 'BFC'
        K.set_session(K.tf.Session(config=cfg))
        
def limit_mem():
    K.get_session().close()
    cfg = K.tf.ConfigProto()
    cfg.gpu_options.allow_growth = True
    K.set_session(K.tf.Session(config=cfg))
    
'''    
def set_keras_backend(backend):

    if K.backend() != backend:
        os.environ['KERAS_BACKEND'] = backend
      
        importlib.reload(K)
        assert K.backend() == backend
    if backend == "tensorflow":
        import tensorflow as tf
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        session = tf.Session(config=config)
 '''       
if __name__ == '__main__':
    #####  create architecture
    set_keras_backend("tensorflow")
    #CHOOSE ONE  - Segmentation or Classification
    #layers=Object_model_2DUNET.my2DUnet()
    layers=Object_model_2DUNET_Class.my2DUnet()
    model=layers.create_DL_Teste5
    #print(model.summary())
    ######  organize files
    #BRAIN CLASS
    files=ManageFilesMulti.myFiles2D(IMAGE_MRI_PATH,IMAGE_MASK_PATH,IMAGE_MASK_BIN_PATH,N_FILES)
    mris,masks=files.openlist()
    mris_training, masks_training, mris_evaluate, masks_evaluate=definetrainevaluate(mris,masks)
    #training
    function_optmizer=definetrainfunctions()
    model, history = compiletrain(mris_training, masks_training, model,function_optmizer)
    evaluate(mris_evaluate, masks_evaluate, model, history)
    print_history_accuracy(history)
    print_history_loss(history)
