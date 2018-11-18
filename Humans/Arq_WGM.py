from keras.layers import BatchNormalization
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, Dropout
from keras.models import *

n=2
f=3 #filter size


class my2DUnet:
    def __init__(self, CLASS_NUMBER, img_rows=256, img_cols=256, img_deep=1):
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.img_deep = img_deep
        self.CLASS_NUMBER=CLASS_NUMBER
        self.inputsize = (img_rows, img_cols, 1)

    @property
    def create_DL(self):
        inputs = Input(self.inputsize)
        conv1 = Conv2D(32 * n, (f, f), activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        conv11 = Conv2D(32 * n, (f, f), activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv11)
        batch1 = BatchNormalization()(pool1)
        drop1 = Dropout(0.1)(batch1)
        conv2 = Conv2D(64 * n, (f, f), activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        conv22 = Conv2D(64 * n, (f, f), activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv22)
        batch2 = BatchNormalization()(pool2)
        drop2 = Dropout(0.1)(batch2)
        conv3 = Conv2D(128 * n, (f, f), activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        conv33 = Conv2D(128 * n, (f, f), activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv33)
        drop3 = Dropout(0.1)(pool3)
        batch3 = BatchNormalization()(drop3)
        conv4 = Conv2D(256 * n, (f, f), activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
        conv44 = Conv2D(256 * n, (f, f), activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
        # drop4 = Dropout(0.1)(conv44)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv44)
        batch4 = BatchNormalization()(pool4)
        conv5 = Conv2D(512 * n, (f, f), activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
        conv55 = Conv2D(512 * n, (f, f), activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
        up6 = Conv2D(256 * n, (2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv55))
        merge6 = concatenate([conv44, up6], axis=3)
        batch6 = BatchNormalization()(merge6)
        drop6 = Dropout(0.1)(batch6)
        conv6 = Conv2D(256 * n, (f, f), activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
        conv66 = Conv2D(256 * n, (f, f), activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
        up7 = Conv2D(128 * n, (2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv66))
        merge7 = concatenate([conv33, up7], axis=3)
        batch7 = BatchNormalization()(merge7)
        drop7 = Dropout(0.1)(batch7)
        conv7 = Conv2D(128 * n, (f, f), activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
        conv77 = Conv2D(128 * n, (f, f), activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
        up8 = Conv2D(64 * n, (2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv77))
        merge8 = concatenate([conv22, up8], axis=3)
        batch8 = BatchNormalization()(merge8)
        drop8 = Dropout(0.1)(batch8)
        conv8 = Conv2D(64 * n, (f, f), activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
        conv88 = Conv2D(64 * n, (f, f), activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
        up9 = Conv2D(32 * n, (2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv88))
        merge9 = concatenate([conv11, up9], axis=3)
        batch9 = BatchNormalization()(merge9)
        drop9 = Dropout(0.1)(batch9)
        conv9 = Conv2D(32 * n, (f, f), activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
        drop5 = Dropout(0.5)(conv9)
        conv99 = Conv2D(32 * n, (f, f), activation='relu', padding='same', kernel_initializer='he_normal')(drop5)
        # conv999 = Conv2D(4, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(conv99)
        conv10 = Conv2D(self.CLASS_NUMBER, (1, 1), activation='softmax')(conv99)
        # drop99 = Dropout(0.1)(conv9)
        # flat=Flatten()(conv99)
        # dense10 = Dense(1, activation='sigmoid')(flat)
        model = Model(input=inputs, output=conv10)
        model.load_weights('/Notebooks/Marianadissertation//Humans/2D-BrainSeg/best_train_models/OLD-WGM/bestweights_vall_loss.hdf5', by_name=True)
        # model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
        print(model.summary())
        return model


'''  def create_DL_H1(self):
        inputs = Input(self.inputsize)
        conv1 = Conv2D(64, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        conv11 = Conv2D(64, (3,3),  activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv11)
        batch1=BatchNormalization()(pool1)
        drop1 = Dropout(0.1)(batch1)
        conv2 = Conv2D(128, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        conv22 = Conv2D(128,(3,3), activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv22)
        batch2=BatchNormalization()(pool2)
        drop2 = Dropout(0.1)(batch2)
        conv3 = Conv2D(256,(3,3), activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        conv33 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv33)
        drop3 = Dropout(0.1)(pool3)
        batch3=BatchNormalization()(drop3)
        conv4 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
        conv44 = Conv2D(512,(3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
        #drop4 = Dropout(0.1)(conv44)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv44)
        batch4=BatchNormalization()(pool4)
        conv5 = Conv2D(1024, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
        conv55 = Conv2D(1024, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
        up6 = Conv2D(512, (2,2), activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv55))
        merge6 = concatenate([conv44, up6], axis=3)
        batch6=BatchNormalization()(merge6)
        drop6 = Dropout(0.1)(batch6)
        conv6 = Conv2D(512,(3,3), activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
        conv66 = Conv2D(512,(3,3), activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
        up7 = Conv2D(256, (2,2), activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv66))
        merge7 = concatenate([conv33 , up7], axis=3)
        batch7=BatchNormalization()(merge7)
        drop7 = Dropout(0.1)(batch7)
        conv7 = Conv2D(256, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
        conv77 = Conv2D(256, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
        up8 = Conv2D(128,( 2,2), activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv77))
        merge8 = concatenate([conv22, up8], axis=3)
        batch8=BatchNormalization()(merge8)
        drop8 = Dropout(0.1)(batch8)
        conv8 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
        conv88 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
        up9 = Conv2D(64,(2,2), activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv88))
        merge9 = concatenate([conv11, up9], axis=3)
        batch9=BatchNormalization()(merge9)
        drop9 = Dropout(0.1)(batch9)
        conv9 = Conv2D(64, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
        #drop5 = Dropout(0.5)(conv9)
        conv99 = Conv2D(64, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        conv10 = Conv2D(1, (1,1), activation='sigmoid')(conv99)
        #drop99 = Dropout(0.1)(conv9)
        #flat=Flatten()(conv99)
        #dense10 = Dense(1, activation='sigmoid')(flat)
        model = Model(input=inputs, output=conv10)
        #model.load_weights('/Notebooks/Marianadissertation/Rats/pythonrats2D/bestvalueslasttrain/Teste4/bestweights_vall_loss.hdf5', by_name=True)
        #model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
        return model



'''
