import numpy as np
from keras.layers import BatchNormalization
from keras.models import *
from keras.layers import Input, concatenate, Conv3D, MaxPooling3D, UpSampling3D, Dropout, Cropping3D
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
n=2
class my3DUnet:
    def __init__(self, img_rows=64, img_cols=64, img_deep=40):
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.img_deep = img_deep
        self.inputsize=(img_rows,img_cols,40,1)
    @property
    def create_DL(self):
        inputs = Input(self.inputsize)
        conv1 = Conv3D(32*n, (3,3,3), activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        conv11 = Conv3D(32*n, (3,3,3),  activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
        pool1 = MaxPooling3D(pool_size=(2,2, 2))(conv11)
        batch1=BatchNormalization()(pool1)
        conv2 = Conv3D(64*n, (3,3,3), activation='relu', padding='same', kernel_initializer='he_normal')(batch1)
        conv22 = Conv3D(64*n,(3,3,3), activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
        pool2 = MaxPooling3D(pool_size=(2,2, 2))(conv22)
        conv3 = Conv3D(128*n,(3,3,3), activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        conv33 = Conv3D(128*n, (3,3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        pool3 = MaxPooling3D(pool_size=(2,2, 2))(conv33)
        drop3 = Dropout(0.5)(conv33)
        conv4 = Conv3D(256*n, (3,3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
        conv44 = Conv3D(256*n,(3,3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
        drop4 = Dropout(0.5)(conv44)
        pool4 = MaxPooling3D(pool_size=(2,2, 1))(drop4)
        conv5 = Conv3D(512*n, (3,3,3), activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
        conv55 = Conv3D(512*n, (3,3,3), activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
        drop5 = Dropout(0.5)(conv55)
        up6 = Conv3D(256*n, (2,2,2), activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling3D(size=(2,2, 1))(drop5))
        merge6 = concatenate([drop4, up6], axis=4)
        conv6 = Conv3D(256*n,(3,3,3), activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
        conv66 = Conv3D(256*n,(3,3,3), activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
        up7 = Conv3D(128*n, (2,2,2), activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling3D(size=(2,2, 2))(conv66))
        merge7 = concatenate([drop3 , up7], axis=4)
        batch7=BatchNormalization()(merge7)
        conv7 = Conv3D(128*n, (3,3,3), activation='relu', padding='same', kernel_initializer='he_normal')(batch7)
        conv77 = Conv3D(128*n, (3,3,3), activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
        up8 = Conv3D(64*n,( 2,2,2), activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling3D(size=(2,2, 2))(conv77))
        merge8 = concatenate([conv22, up8], axis=4)
        conv8 = Conv3D(64*n, (3,3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
        conv88 = Conv3D(64*n, (3,3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
        batch8=BatchNormalization()(conv88)
        up9 = Conv3D(32*n,(2,2,2), activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling3D(size=(2,2, 2))(batch8))
        merge9 = concatenate([conv11, up9], axis=4)
        conv9 = Conv3D(32*n, (3,3,3), activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
        conv99 = Conv3D(32*n, (3,3,3), activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        conv10 = Conv3D(1, (1,1,1), activation='sigmoid')(conv99)
        model = Model(input=inputs, output=conv10)
        print(model.summary())
        return model


