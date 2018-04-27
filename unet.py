import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
from keras.models import *
from keras.layers import Input, merge, concatenate, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D, core, Conv2DTranspose
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from data import *
#from opt import *
from losses import *
from keras.models import load_model

# Global vars for easy access and who cares about programming principles.
_epochs = 10
_which_model = 5 # 5=ultra.
_batch_size = 8
_lr = 1e-06
_tensor_in = "N/A"
_test_shape = "N/A"


class myUnet(object):

	# set to img size
    def __init__(self, img_rows=256, img_cols=256):

        self.img_rows = img_rows
        self.img_cols = img_cols
        smooth = 1.0

    def load_data(self):

        mydata = dataProcess(self.img_rows, self.img_cols) # in data.py there's a class dataProcess
        imgs_train, imgs_mask_train = mydata.load_train_data()
        imgs_test = mydata.load_test_data()
        return imgs_train, imgs_mask_train, imgs_test

    def dice_coef(y_true, y_pred):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    
    def dice_coef_loss(y_true, y_pred):
        return -dice_coef(y_true, y_pred)

    def get_unet_hao(self):
        img_in = Input((self.img_rows, self.img_cols, 1))
        
        feature_amount = 32
        conv1 = Conv2D(feature_amount, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(img_in)
        conv1 = Conv2D(feature_amount, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        # ---
        conv2 = Conv2D(feature_amount*2, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        conv2 = Conv2D(feature_amount*2, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        # ---
        conv3 = Conv2D(feature_amount*4, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        conv3 = Conv2D(feature_amount*4, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        # ---
        conv4 = Conv2D(feature_amount*8, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
        conv4 = Conv2D(feature_amount*8, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        # --- lowest layer.
        conv5 = Conv2D(feature_amount*16, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
        conv5 = Conv2D(feature_amount*16, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
        # --- and back up.

        # Upsample previous convo layer.
        up5       = UpSampling2D(size=(2, 2))(conv5)
        # 2x2-Convolute the upsample.
        conv_up5  = Conv2D(feature_amount*8, (2,2), activation='relu', padding='same', kernel_initializer='he_normal')(up5)
        # Concatenate convoluted upsampled convo with same sized convo from down path.
        concat5_4 = concatenate([conv_up5, conv4], axis=3)
        # 2 3x3 convos
        conv6     = Conv2D(feature_amount*8, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(concat5_4)
        conv6     = Conv2D(feature_amount*8, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
        # ---
        up6       = UpSampling2D(size=(2, 2))(conv6)
        conv_up6  = Conv2D(feature_amount*4, (2,2), activation='relu', padding='same', kernel_initializer='he_normal')(up6)
        concat6_3 = concatenate([conv_up6, conv3], axis=3)
        conv7     = Conv2D(feature_amount*4, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(concat6_3)
        conv7     = Conv2D(feature_amount*4, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
        # ---
        up7       = UpSampling2D(size=(2, 2))(conv7)
        conv_up7  = Conv2D(feature_amount*2, (2,2), activation='relu', padding='same', kernel_initializer='he_normal')(up7)
        concat7_2 = concatenate([conv_up7, conv2], axis=3)
        conv8     = Conv2D(feature_amount*2, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(concat7_2)
        conv8     = Conv2D(feature_amount*2, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
        # ---
        up8       = UpSampling2D(size=(2, 2))(conv8)
        conv_up8  = Conv2D(feature_amount, (2,2), activation='relu', padding='same', kernel_initializer='he_normal')(up8)
        concat8_1 = concatenate([conv_up8, conv1], axis=3)
        conv9     = Conv2D(feature_amount, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(concat8_1)
        conv9     = Conv2D(feature_amount, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(conv9)

        img_out   = Conv2D(1, (1,1), activation='sigmoid')(conv9)

        # Get Model ready and compile.
        model = Model(input=img_in, output=img_out)
        lr = 1e-08
        loss = dice_coef_loss
        accuracy = 'accuracy'
        model.compile(optimizer=Adam(lr=lr),
                      loss=loss,
                      metrics=[accuracy])
        print("=== Model Specs: LR: " + str(lr) + " === Loss: " + str(loss) + " === Metric: " + str(accuracy) + " ===")
        return model

        # End of function.
        
    def get_unet2(self):

        input1 = Input((self.img_rows, self.img_cols, 1))
        
        feature_amount = 32
        conv1 = Conv2D(feature_amount, 3, activation='relu', padding='same', kernel_initializer='he_normal')(input1)
        conv1 = Conv2D(feature_amount, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(feature_amount*2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        conv2 = Conv2D(feature_amount*2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(feature_amount*4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        conv3 = Conv2D(feature_amount*4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        # =============
        conv4 = Conv2D(feature_amount*8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
        conv4 = Conv2D(feature_amount*8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
        drop4 = Dropout(0.5)(conv4)
        #pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        #conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
        #conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
        #drop5 = Dropout(0.5)(conv5)

        #up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(drop5))
        #merge6 = concatenate([drop4, up6], axis=3)
        #conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
        #conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

        # ============
        up7 = Conv2D(feature_amount*4, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(drop4))
        merge7 = concatenate([conv3, up7], axis=3)
        conv7 = Conv2D(feature_amount*4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
        conv7 = Conv2D(feature_amount*4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

        up8 = Conv2D(feature_amount*2, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv7))
        merge8 = concatenate([conv2, up8], axis=3)
        conv8 = Conv2D(feature_amount*2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
        conv8 = Conv2D(feature_amount*2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

        up9 = Conv2D(feature_amount, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8))
        merge9 = concatenate([conv1, up9], axis=3)
        conv9 = Conv2D(feature_amount, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
        conv9 = Conv2D(feature_amount, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)
        
    

        model = Model(inputs=input1, outputs=conv10)
        #model = Model(inputs=input1, outputs=conv9)
        #asdf = dice_coef(y_true, y_pred)
        #dice_fn = dicee()


        #dloss = dice_coef()
        #d     = dice_coef_loss()

        
        lr = 1e-07
        print(str(lr) + " Learning rate.")
        loss = 1
        if loss==0:
            print("Loss: binary_crossentropy.")
            model.compile(optimizer=Adam(lr=lr),
                          loss=['binary_crossentropy'],
                          metrics=['accuracy'])
                          #sample_weight_mode='temporal') 
        elif loss==1:
            print("Loss: dice_coef_loss.")
            model.compile(optimizer=Adam(lr=lr), 
                          loss=dice_coef_loss,
                          metrics=['accuracy'])

        return model

    def get_unet3(self):

        print("HELLO?")
        input1 = Input((self.img_rows, self.img_cols, 1)) #, 1)) #MondayNightTODO
        
        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(input1)
        conv1 = Dropout(0.2)(conv1)
        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)

        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        #
        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        conv2 = Dropout(0.2)(conv2)
        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
        
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        #
        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        conv3 = Dropout(0.2)(conv3)
        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)

        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        #
        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
        conv4 = Dropout(0.2)(conv4)
        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
        
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
        #
        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
        conv5 = Dropout(0.2)(conv5)
        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
        

        # The green arrow.
        up6    = UpSampling2D(size=(2, 2))(conv5)
        #up6    = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(up6)
        # The grey arrow.
        merge6 = concatenate([conv4, up6], axis=3)
        # 2 times purple arrow.
        conv6  = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
        conv6 = Dropout(0.2)(conv6)
        conv6  = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)


        #
        up7    = UpSampling2D(size=(2, 2))(conv6)
        #up7    = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(up7)
        merge7 = concatenate([conv3, up7], axis=3)
        conv7  = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
        conv7  = Dropout(0.2)(conv7)
        conv7  = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
        #
        up8    = UpSampling2D(size=(2, 2))(conv7)
        #up8    = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(up8)
        merge8 = concatenate([conv2, up8], axis=3)
        conv8  = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
        conv8  = Dropout(0.2)(conv8)
        conv8  = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
        #
        up9    = UpSampling2D(size=(2, 2))(conv8)
        #up9    = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(up9)
        merge9 = concatenate([conv1, up9], axis=3)
        conv9  = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
        conv9  = Dropout(0.2)(conv9)
        conv9  = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        #
        conv9  = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        conv10 = Conv2D(1, 1, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        ##conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)
        
        #conv10 = Conv2D(2, 1, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        #conv10 = core.Reshape((2, self.img_rows*self.img_cols))(conv10)
        #conv10 = core.Permute((2,1))(conv10)

        #print("conv10.shape: " + str(conv10.shape))
        #conv11 = core.Activation('softmax')(conv10)

    

        model = Model(inputs=input1, outputs=conv10)

        lr = 1e-07
        print(str(lr) + " Learning rate.")
        loss = 1
        if loss==0:
            print("Loss: binary_crossentropy.")
            model.compile(optimizer=Adam(lr=lr),
                          loss=['binary_crossentropy'],
                          metrics=['accuracy'])
                          #sample_weight_mode='temporal') 
        elif loss==1:
            print("Loss: dice_coef_loss.")
            model.compile(optimizer=Adam(lr=lr), 
                          loss=dice_coef_loss,
                          metrics=['accuracy'])
        elif loss==2:
            print("Loss: categorical_crossentropy.")
            sgd = SGD(lr=0.01, decay=1e-6, momentum=0.3, nesterov=False)
            model.compile(optimizer=sgd,
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])

        return model
        # End Unet3.

    def get_unet(self):

        input1 = Input((self.img_rows, self.img_cols, 1))
        
        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(input1)
        print("conv1 shape:", conv1.shape)
        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
        print("conv1 shape:", conv1.shape)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        print("pool1 shape:", pool1.shape)

        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        print("conv2 shape:", conv2.shape)
        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
        print("conv2 shape:", conv2.shape)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        print("pool2 shape:", pool2.shape)

        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        print("conv3 shape:", conv3.shape)
        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        print("conv3 shape:", conv3.shape)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        print("pool3 shape:", pool3.shape)

        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
        print("conv4 shape:", conv4.shape)
        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
        print("conv4 shape:", conv4.shape)
        drop4 = Dropout(0.5)(conv4)
        print("drop4 shape:", drop4.shape)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
        print("pool4 shape:", pool4.shape)

        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
        print("conv5 shape:", conv5.shape)
        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
        print("conv5 shape:", conv5.shape)
        drop5 = Dropout(0.5)(conv5)
        print("drop5 shape:", drop5.shape)

        up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(drop5))
        print("up6 shape:", up6.shape)
        merge6 = concatenate([drop4, up6], axis=3)
        print("merge6 shape:", merge6.shape)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
        print("conv6 shape:", conv6.shape)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
        print("conv6 shape:", conv6.shape)

        up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv6))
        print("up7 shape:", up7.shape)
        merge7 = concatenate([conv3, up7], axis=3)
        print("merge7 shape:", merge7.shape)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
        print("conv7 shape:", conv7.shape)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
        print("conv7 shape:", conv7.shape)

        up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv7))
        print("up8 shape:", up8.shape)
        merge8 = concatenate([conv2, up8], axis=3)
        print("merge8 shape:", merge8.shape)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
        print("conv8 shape:", conv8.shape)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
        print("conv8 shape:", conv8.shape)

        up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8))
        print("up9 shape:", up9.shape)
        merge9 = concatenate([conv1, up9], axis=3)
        print("merge9 shape:", merge9.shape)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
        print("conv9 shape:", conv9.shape)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        print("conv9 shape:", conv9.shape)
        conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        print("conv9 shape:", conv9.shape)
        conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)
        print("conv10 shape:", conv10.shape)
        
	

        model = Model(inputs=input1, outputs=conv10)

        
        lr = 1e-08
        print(str(lr) + " Learning rate.")
        loss = 1
        if loss==0:
            print("Loss: binary_crossentropy.")
            model.compile(optimizer=Adam(lr=lr),
                          loss=['binary_crossentropy'],
                          metrics=['accuracy'])
                          #sample_weight_mode='temporal') 
        elif loss==1:
            print("Loss: dice_coef_loss.")
            model.compile(optimizer=Adam(lr=lr), 
                          loss=dice_coef_loss,
                          metrics=['accuracy'])

        return model

    def get_unet_ultra(self):
        # https://github.com/jocicmarko/ultrasound-nerve-segmentation/blob/master/train.py
        inputs = Input((self.img_rows, self.img_cols, 1))
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
        conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

        up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
        conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
        conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

        up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
        conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
        conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

        up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
        conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
        conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

        up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
        conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
        conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

        conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

        model = Model(inputs=[inputs], outputs=[conv10])
        lr = _lr
        print("Learning Rate: " + str(lr))
        model.compile(optimizer=Adam(lr=lr), loss=dice_coef_loss, metrics=[dice_coef])

        return model


    def train(self, epochs, which_model, batch_size):

        print("Loading data for training and testing...")#, end="")
        imgs_train, imgs_mask_train, imgs_test = self.load_data()
        print("... done.")
        global _tensor_in, _test_shape
        _tensor_in = imgs_train.shape
        _test_shape = imgs_test.shape

        print("Getting model...")
        load_prev_model = which_model
        if load_prev_model == 0:
            model = load_model('unet.hdf5', custom_objects={'dice_coef_loss': dice_coef_loss})
            print("Loaded model from hdf5.")

        elif load_prev_model == 1:
            model = self.get_unet()
            print("Created orig. Unet.")

        elif load_prev_model == 2:
            model = self.get_unet2()
            print("Created smaller Unet.")

        elif load_prev_model == 3:
            #imgs_mask_train = core.Reshape((2, self.img_rows*self.img_cols))(imgs_mask_train)
            #imgs_mask_train = core.Permute((2,1))(imgs_mask_train)
            model = self.get_unet3()
            print("Created Unet3.")

        elif load_prev_model == 4:
            model = self.get_unet_hao()
            print("Created Unet_hao.")
        elif load_prev_model == 5:
            model = self.get_unet_ultra()
            print("Created Unet Ultrasound.")
        

        model_checkpoint = ModelCheckpoint('unet.hdf5', 
                                            monitor='loss', 
                                            verbose=1, 
                                            save_best_only=False)
        
        
        # 1436 images
        #class_weights = np.zeros((1436, 102400))
        #class_weights[:, 0] += 0.05
        #class_weights[:, 1] += 0.95

        print('Fitting model...')
        # LET'S EXPERIMENT WITH DATA AUGMENTATION YAY. This does not work. TODO
        augment = False
        if augment==True:
            data_gen_args = dict(featurewise_center=False, #TODO do i need or not? is it all data or just generated data?
                                 featurewise_std_normalization=False,
                                 rotation_range=20.)#,
                                 #width_shift_range=0.1,
                                 #height_shift_range=0.1,
                                 #zoom_range=0.15)
            image_datagen = ImageDataGenerator(**data_gen_args)
            mask_datagen = ImageDataGenerator(**data_gen_args)

            # Provide the same seed and keyword arguments to the fit and flow methods
            seed = 1
            image_datagen.fit(imgs_train, augment=True, seed=seed)
            mask_datagen.fit(imgs_mask_train, augment=True, seed=seed)

            image_generator = image_datagen.flow(imgs_train,
                                                 seed=seed)#,
                                                 #save_to_dir="./augment/img/")
            mask_generator = mask_datagen.flow(imgs_mask_train,
                                               seed=seed)#,
                                               #save_to_dir="./augment/mask/")

            # combine generators into one which yields image and masks
            train_generator = zip(image_generator, mask_generator) #todo

            # END DATA AUGMENTATION SADFACE.

        epochs = _epochs
        batch_size = _batch_size
        #augment git gen function instead of fit
        if augment==True:
            model.fit_generator(train_generator,
                                steps_per_epoch=1395/batch_size,
                                epochs=epochs,
                                verbose=1)#,
                                #max_queue_size=10)
        else:
            print("model.fit input: imgs_train.shape: " + str(imgs_train.shape))
            print("model.fit input: imgs_mask_train.shape: " + str(imgs_mask_train.shape))
            model.fit(imgs_train, 
                      imgs_mask_train, 
                      batch_size=batch_size, 
                      epochs=epochs, 
                      verbose=1, 
                      validation_split=0.1, 
                      shuffle=True, 
                      callbacks=[model_checkpoint])#,
                #sample_weight = class_weights)
                    #, class_weight = {0:1, 1:100}

        print('Predicting test data...')
        imgs_mask_test = model.predict(imgs_test, batch_size=1, verbose=1)
        np.save('./results/imgs_mask_test.npy', imgs_mask_test)
        print('... done.')
        #pred_test_data(imgs_test) # Predict and write to numpy file.
        
        # End of function.

    def pred_test_data():
        print('Predicting test data...')
        imgs_mask_test = model.predict(imgs_test, batch_size=1, verbose=1)
        np.save('./results/imgs_mask_test.npy', imgs_mask_test)
        print('... done.')

        # End of function.

    def save_img(self):

        print("array to image")
        imgs = np.load('./results/imgs_mask_test.npy')
        print("test result imgs Max: " + str(np.max(np.array(imgs))))
        print("test result imgs Min: " + str(np.min(np.array(imgs))))
        #print("shape should be 320x320xsomething" + str(imgs.shape)) # Seems to be correct.
        #print("should equal amount of images: " + str(imgs.shape[0]))
        for i in range(imgs.shape[0]):
            img = imgs[i]
            if np.isnan(np.sum(img)):
                print("nan - not a number?")

            img = array_to_img(img)
            #print(img.size)
            img.save("./results/%d.tif" % (i))

    def save_model_info(self):
        file = open("./results/model_info.txt","w")
        file.write("epochs: " + str(_epochs) + "\n")
        file.write("which_model: " + str(_which_model) + "\n")
        file.write("_batch_size: " + str(_batch_size) + "\n")
        file.write("_lr: " + str(_lr) + "\n")
        file.write("_tensor_in: " + str(_tensor_in) + "\n")
        file.write("_tensor_in: " + str(_tensor_in) + "\n")
        file.close()


if __name__ == '__main__':
    myunet = myUnet()
    #train(epochs, which_model, batch_size):
    myunet.train(10, 5, 8)
    myunet.save_img()
    save_model_info()

    #for run_number in range(1,11):
    #    myunet.train(1, 0, 8) #0 = load from hdf5
    #    myunet.save_img()
    #    print("Go check output images." * 5 + "Epochs done: " + str(run_number+1))
_epochs = 10
_which_model = 5 # 5=ultra.
_batch_size = 8
_lr = 1e-06
_tensor_in = "N/A"
_test_shape = "N/A"