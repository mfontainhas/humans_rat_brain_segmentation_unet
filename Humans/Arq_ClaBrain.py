from keras.models import *
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D,Conv2DTranspose,BatchNormalization, Dropout,Flatten,Dense
from keras.utils.vis_utils import plot_model
n=1 
f=3 #filter size
class my2DUnet:
    def __init__(self, img_rows=256, img_cols=256, img_deep=1):
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.img_deep = img_deep
        self.inputsize=(img_rows,img_cols,1)
    @property

    def create_DL_C4:
        model = Sequential()
        model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(256, 256, 1)))
        model.add(BatchNormalization(axis=1))
        model.add(Conv2D(64, (3, 3), activation='relu',kernel_initializer='random_uniform'))
        model.add(MaxPooling2D((3, 3), strides=(3, 3)))
        model.add(Dropout(0.3))
        model.add(BatchNormalization())
        model.add(Conv2D(128, (3, 3), activation='relu',kernel_initializer='random_uniform'))
        model.add(MaxPooling2D((3,3), strides=(3,3)))
        model.add(Conv2D(192, (3, 3), activation='relu',kernel_initializer='random_uniform'))
        model.add(MaxPooling2D((3, 3), strides=(3, 3)))
        model.add(Dropout(0.3))
        model.add(BatchNormalization())
        model.add(Conv2D(256, (3, 3), activation='relu',kernel_initializer='random_uniform'))
        model.add(MaxPooling2D((3, 3), strides=(3, 3)))
        model.add(BatchNormalization(axis=1))

        model.add(Dropout(0.3))
        model.add(Flatten())
        model.add(Dense(516, activation='relu'))
        model.add(Dropout(0.4))
        model.add(BatchNormalization(axis=1))
        model.add(Dense(516, activation='relu'))
        model.add(Dropout(0.4))
        model.add(Dense(1, activation='sigmoid'))
        
        print(model.summary())
        plot_model(model,show_shapes=True, to_file='/Notebooks/Marianadissertation/Humans/modelC4.jpg')
        return model

    def create_DL_C3(weights_path=None):
        model = Sequential()
        model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(256, 256, 1)))
        model.add(BatchNormalization(axis=1))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D((3, 3), strides=(3, 3)))
        model.add(Dropout(0.3))
        model.add(BatchNormalization())
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D((3, 3), strides=(3, 3)))
        model.add(Conv2D(192, (3, 3), activation='relu'))
        model.add(MaxPooling2D((3, 3), strides=(3, 3)))
        model.add(Dropout(0.3))
        model.add(BatchNormalization())
        model.add(Conv2D(256, (3, 3), activation='relu'))
        model.add(MaxPooling2D((3, 3), strides=(3, 3)))
        model.add(BatchNormalization(axis=1))

        model.add(Dropout(0.3))
        model.add(Flatten())
        model.add(Dense(516, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(516, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='sigmoid'))

        print(model.summary())
        return model

    def create_DL_C2( weights_path=None):
        model = Sequential()
        model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(256, 256, 1)))
        model.add(BatchNormalization(axis=1))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D((3, 3), strides=(3, 3)))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())
        model.add(Conv2D(128, (3, 3), activation='relu'))
        #model.add(MaxPooling2D((3,3), strides=(3,3)))
        model.add(Conv2D(192, (3, 3), activation='relu'))
        model.add(MaxPooling2D((3, 3), strides=(3, 3)))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())
        model.add(Conv2D(256, (3, 3), activation='relu'))
        model.add(MaxPooling2D((3, 3), strides=(3, 3)))
        model.add(BatchNormalization(axis=1))

        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(516, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(516, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='sigmoid'))

        print(model.summary())
        return model
    
    def create_DL_C1( weights_path=None):
        model = Sequential()
        model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(256, 256, 1)))
        model.add(BatchNormalization(axis=1))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D((3, 3), strides=(3, 3)))
        model.add(BatchNormalization())
        model.add(Conv2D(128, (3, 3), activation='relu'))
        #model.add(MaxPooling2D((3,3), strides=(3,3)))
        model.add(Conv2D(192, (3, 3), activation='relu'))
        model.add(MaxPooling2D((3, 3), strides=(3, 3)))
        model.add(BatchNormalization())
        model.add(Conv2D(256, (3, 3), activation='relu'))
        model.add(MaxPooling2D((3, 3), strides=(3, 3)))
        model.add(BatchNormalization(axis=1))

        #model.add(Dropout(0.6))
        model.add(Flatten())
        model.add(Dense(516, activation='relu'))
        #model.add(Dropout(0.7))
        model.add(Dense(516, activation='relu'))
        #model.add(Dropout(0.7))
        model.add(Dense(1, activation='sigmoid'))

        print(model.summary())
        return model
    
if __name__ == '__main__':
    layers = my2DUnet()
    print("HEY")
    model = layers.create_DL_C4()