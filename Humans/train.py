import importlib
import os
import time
from keras import losses, optimizers
import Arq_ClaBrain,Arq_SegBrain,Arq_WGM
import Managefiles_classification, Managefiles_mask, Managefiles_wgm,Managefiles_subc
import loss_functions
import psutil
import writegraphs
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping#LR

##################  BEGIN VARIABLES  #################################
# Prepare files info
TYPEOFDEEPLEARNING=2 # 0 - classification of brain VS no brain ; 1 - Segment only brain; 2- Segment white and gray matter
CLASS_NUMBER=3
N_FILES = 54  # Max number of files = 54
IMAGE_PATH = '/Notebooks/Marianadissertation/Humans/T1_WGM_ASEG'
MODEL_TRAIN_SAVE = '/Notebooks/Marianadissertation/Humans/2D-BrainSeg/trainingVisualize/models'
MODEL_SAVE_BEST = '/Notebooks/Marianadissertation/Humans/2D-BrainSeg/best_train_models'
MODEL_FINAL_SAVE = '/Notebooks/Marianadissertation/Humans/2D-BrainSeg/logs/final_'
SPLIT_PERCENTAGE = 1 # Define percentage of Training images, the rest will be evaluation#
# Training info
EPOCHS = 1000
BATCH_SIZE = 5
# Define parameters of optmizer
lrate = 0.0003
decay = lrate / 200
momentum = 0.9
VERBOSE_TRAIN = 0.9  # 0-only text, 1-generating images
# See images (not working currently)
VERBOSE_imageGenerator = 1  # 0-no train comments, 1-text each epoch
PRINT_NUMBER = 20  # Choose a number to print in a multiple epoch
SELECTED_LAYERS = [3, 6, 10, 20, 30, 35, 37]  # Select only some Layers to print featuremaps
#################  ENDING VARIABLES  #################################

def definetrainevaluate(mris, masks):  # split train data and evaluate data
    split_len = int(len(mris) * SPLIT_PERCENTAGE)
    mris_training = mris[:split_len]
    masks_training = masks[:split_len]
    mris_evaluate = mris[split_len:]
    masks_evaluate = masks[split_len:]
    return mris_training, masks_training, mris_evaluate, masks_evaluate


def compiletrain(mris_training, masks_training, model):
    print("\n ########### TRAINING ############\n")##
    # define metrics
    
    if TYPEOFDEEPLEARNING==0:
        all_metrics = ['accuracy', loss_functions.Tversky_coef_loss, loss_functions.Precision, loss_functions.FPR,
                       loss_functions.FNR, loss_functions.specificity, loss_functions.sensitivity]
        model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=all_metrics)
    elif TYPEOFDEEPLEARNING==1:
        all_metrics = ['accuracy', loss_functions.Precision, loss_functions.FPR,
                       loss_functions.FNR, loss_functions.specificity]
        model.compile(loss=loss_functions.dice_coef_loss, optimizer='sgd', metrics=all_metrics)
    elif TYPEOFDEEPLEARNING==2:
        all_metrics = ['accuracy', loss_functions.sensitivity_background, loss_functions.sensitivity_tissue1, loss_functions.sensitivity_tissue2, loss_functions.dice_coef_multilabel]
        sgd = optimizers.SGD(lr=lrate, momentum=0.8, decay=decay, nesterov=False)
        model.compile(loss=loss_functions.dice_coef_multilabel, optimizer=sgd, metrics=all_metrics)
    elif TYPEOFDEEPLEARNING==3:
        all_metrics = ['accuracy',loss_functions.sensitivity_background, loss_functions.sensitivity_tissue1, loss_functions.sensitivity_tissue2, loss_functions.sensitivity_tissue3,loss_functions.sensitivity_tissue4,loss_functions.dice_coef_multilabel,loss_functions.Tversky_multilabel]
        sgd = optimizers.SGD(lr=lrate, momentum=0.8, decay=decay, nesterov=False)
        model.compile(loss=loss_functions.assimetric_loss_multilabel, optimizer=sgd, metrics=all_metrics)

    # Save tensorboard
    tensorboard = writegraphs.TrainValTensorBoard(write_graph=False)
    # Save best models
    m1 = ModelCheckpoint(MODEL_SAVE_BEST + "/val_loss.h5", monitor='val_loss', verbose=0, save_best_only=True,
                         save_weights_only=False, mode='auto', period=1)
    m2 = ModelCheckpoint(MODEL_SAVE_BEST + "/val_acc.h5", monitor='val_acc', verbose=0, save_best_only=True,
                         save_weights_only=False, mode='auto', period=1)
    m3 = ModelCheckpoint(MODEL_SAVE_BEST + "/bestweights_vall_acc.hdf5", monitor='val_acc', verbose=0,
                         save_best_only=True, save_weights_only=True, mode='auto', period=1)
    m4 = ModelCheckpoint(MODEL_SAVE_BEST + "/bestweights_vall_loss.hdf5", monitor='val_loss', verbose=0,
                         save_best_only=True, save_weights_only=True, mode='auto', period=1)
    # Early Stoppping#
    e1 = EarlyStopping(monitor='val_loss', min_delta=0, patience=80, verbose=0, mode='auto')
    e2 = EarlyStopping(monitor='val_acc', min_delta=0, patience=80, verbose=0, mode='auto')
    # Fit model with images, begin training
    history = model.fit(mris_training, masks_training, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1,
                        validation_split=0.2, initial_epoch=0, callbacks=[tensorboard, m1, m2, m3, m4, e1, e2],
                        shuffle=True)
    # save final Model
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    model.save(MODEL_FINAL_SAVE + timestamp + ".h5")
    return model, history


def evaluate(mris_evaluate, masks_evaluate, model, history):
    p = psutil.Process()
    cpu_time = p.cpu_times()[0]
    mem = p.memory_info()[0]
    print("###########Evaluation############\n")
    scores = model.evaluate(mris_evaluate, masks_evaluate)
    print("\n Metric: %s: %.2f%%\n" % (model.metrics_names[1], scores[1] * 100))
    with open("/Notebooks/Marianadissertation/Rats/pythonrats2/log.csv", "a") as myfile:
        myfile.write(",")
        myfile.write("\"" + ascii(model).strip() + "\",")
        myfile.write("{},".format(history.history['loss'][-1]))
        myfile.write("{},".format(history.history['acc'][-1]))
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
        # cfg.gpu_options.allocator_type = 'BFC'
        K.set_session(K.tf.Session(config=cfg))


def limit_mem():
    K.get_session().close()
    cfg = K.tf.ConfigProto()
    cfg.gpu_options.allow_growth = True
    K.set_session(K.tf.Session(config=cfg))


if __name__ == '__main__':
    ##############################    SET BACKGROUND KERAS DEEP LEARNING   ##############################
    set_keras_backend("tensorflow")
    ##############################     PREPARE FILES FOR DEEP LEARNING     ##############################
    # How many files exist?
    files_list = sorted(os.listdir(IMAGE_PATH))
    print("There are " + str(int(len(files_list) / 4)) + " patients. \n")
    # organize files
    # Managefiles2D_wmg or Managefiles2D_mask

    ##############################   DEFINE DEEP LEARNING ARCHITECTURE    ##############################
    #####################################   AND PREPARE IMAGES    ######################################
    
    if TYPEOFDEEPLEARNING == 0:  # Classify layers brain vs no brain
        files = Managefiles_classification.myFiles2D(IMAGE_PATH, N_FILES)
        mris, masks = files.openlist()
        mris_training, masks_training, mris_evaluate, masks_evaluate = definetrainevaluate(mris, masks)
        layers = Arq_ClaBrain.my2DUnet()
        model = layers.create_DL_C4()
        print(model.summary())

    elif TYPEOFDEEPLEARNING == 1:  # Segment only brain
        files = Managefiles_mask.myFiles2D(IMAGE_PATH, N_FILES)
        mris, masks = files.openlist()
        mris_training, masks_training, mris_evaluate, masks_evaluate = definetrainevaluate(mris, masks)
        layer=Arq_SegBrain.my2DUnet()
        model=layer.create_DL
        print(model.summary())

    elif TYPEOFDEEPLEARNING == 2:  # Segment White and Gray Matter
        files = Managefiles_wgm.myFiles2D(IMAGE_PATH, N_FILES,CLASS_NUMBER)
        mris, masks = files.openlist()
        mris_training, masks_training, mris_evaluate, masks_evaluate = definetrainevaluate(mris, masks)
        layer = Arq_WGM.my2DUnet(CLASS_NUMBER)
        model = layer.create_DL
        
    elif TYPEOFDEEPLEARNING == 3:  # Segment Subcortical
        files = Managefiles_subc.myFiles2D(IMAGE_PATH, N_FILES,CLASS_NUMBER)
        mris, masks = files.openlist()
        mris_training, masks_training, mris_evaluate, masks_evaluate = definetrainevaluate(mris, masks)
        layer = Arq_WGM.my2DUnet(CLASS_NUMBER)
        model = layer.create_DL

    ################################         TRAINING DEEP LEARNING         ##############################
    model, history = compiletrain(mris_training, masks_training, model)
    evaluate(mris_evaluate, masks_evaluate, model, history)
    print_history_accuracy(history)
    print_history_loss(history)
