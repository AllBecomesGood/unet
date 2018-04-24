import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
from keras.models import *
from keras.layers import Input, merge, concatenate, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from data import *
#from opt import *
from losses import *
from keras.models import load_model


class myUnet(object):

	# set to img size
    def __init__(self, img_rows=320, img_cols=320):

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

    def get_unet(self):

        input1 = Input((self.img_rows, self.img_cols, 1)) #, 1)) #MondayNightTODO
        
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
        #model = Model(inputs=input1, outputs=conv9)
	    #asdf = dice_coef(y_true, y_pred)
        #dice_fn = dicee()


        #dloss = dice_coef()
        #d     = dice_coef_loss()

        

        model.compile(optimizer=Adam(lr=0.001),#Adam(lr=1e-5), 
                    loss=['binary_crossentropy'], #, dice_coef_loss
                    metrics=['accuracy'])#[dice_coef])#,
                    #sample_weight_mode='temporal') #loss=dice_coef_loss, 

        return model

    
    #def dice_coef(y_true, y_pred):
    #    y_true_f = keras.flatten(y_true)
     #   y_pred_f = keras.flatten(y_pred)
      #  intersection = keras.sum(y_true_f * y_pred_f)
       # coef = (2. * intersection + keras.epsilon()) / (keras.sum(y_true_f) + keras.sum(y_pred_f) + keras.epsilon())
        #return coef


    def train(self):

        print("loading data")
        imgs_train, imgs_mask_train, imgs_test = self.load_data()
        print("loading data done")
        #model = self.get_unet()
        model = load_model('unet.hdf5')
        print("got unet")

        model_checkpoint = ModelCheckpoint('unet.hdf5', 
                                            monitor='loss', 
                                            verbose=1, 
                                            save_best_only=False)
        print('Fitting model...')
        
        # 1436 images
        #class_weights = np.zeros((1436, 102400))
        #class_weights[:, 0] += 0.05
        #class_weights[:, 1] += 0.95

        

        model.fit(imgs_train, 
                imgs_mask_train, 
                batch_size=8, 
                epochs=20, 
                verbose=1, 
                validation_split=0.1, 
                shuffle=True, 
                callbacks=[model_checkpoint])#,
                #sample_weight = class_weights)

                    #, class_weight = {0:1, 1:100}

        print('predict test data')
        imgs_mask_test = model.predict(imgs_test, batch_size=1, verbose=1)
        np.save('./results/imgs_mask_test.npy', imgs_mask_test)

    def save_img(self):

        print("array to image")
        imgs = np.load('./results/imgs_mask_test.npy')
        #print("shape should be 320x320xsomething" + str(imgs.shape)) # Seems to be correct.
        #print("should equal amount of images: " + str(imgs.shape[0]))
        for i in range(imgs.shape[0]):
            img = imgs[i]
            if i == 30:
                print("img in unet.py Max: " + str(np.max(np.array(img))))
                print("img in unet.py Min: " + str(np.min(np.array(img))))
                print(img.shape)
            if np.isnan(np.sum(img)):
                print("NANI?")

            img = array_to_img(img)
            #print(img.size)
            img.save("./results/%d.tif" % (i))


if __name__ == '__main__':
    myunet = myUnet()
    myunet.train()
    myunet.save_img()
