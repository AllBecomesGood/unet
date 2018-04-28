import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
from keras.models import *
from keras.layers import Input, merge, MaxPooling2D, UpSampling2D, Dropout, Cropping2D, Convolution2D, BatchNormalization
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

    # def get_unet(self):

    #     input1 = Input((self.img_rows, self.img_cols, 1)) #, 1)) #MondayNightTODO
        
    #     conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(input1)
    #     print("conv1 shape:", conv1.shape)
    #     conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    #     print("conv1 shape:", conv1.shape)
    #     pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    #     print("pool1 shape:", pool1.shape)

    #     conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    #     print("conv2 shape:", conv2.shape)
    #     conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    #     print("conv2 shape:", conv2.shape)
    #     pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    #     print("pool2 shape:", pool2.shape)

    #     conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    #     print("conv3 shape:", conv3.shape)
    #     conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    #     print("conv3 shape:", conv3.shape)
    #     pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    #     print("pool3 shape:", pool3.shape)

    #     conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    #     print("conv4 shape:", conv4.shape)
    #     conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    #     print("conv4 shape:", conv4.shape)
    #     drop4 = Dropout(0.5)(conv4)
    #     print("drop4 shape:", drop4.shape)
    #     pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    #     print("pool4 shape:", pool4.shape)

    #     conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    #     print("conv5 shape:", conv5.shape)
    #     conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    #     print("conv5 shape:", conv5.shape)
    #     drop5 = Dropout(0.5)(conv5)
    #     print("drop5 shape:", drop5.shape)

    #     up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(drop5))
    #     print("up6 shape:", up6.shape)
    #     merge6 = concatenate([drop4, up6], axis=3)
    #     print("merge6 shape:", merge6.shape)
    #     conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    #     print("conv6 shape:", conv6.shape)
    #     conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
    #     print("conv6 shape:", conv6.shape)

    #     up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv6))
    #     print("up7 shape:", up7.shape)
    #     merge7 = concatenate([conv3, up7], axis=3)
    #     print("merge7 shape:", merge7.shape)
    #     conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    #     print("conv7 shape:", conv7.shape)
    #     conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
    #     print("conv7 shape:", conv7.shape)

    #     up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv7))
    #     print("up8 shape:", up8.shape)
    #     merge8 = concatenate([conv2, up8], axis=3)
    #     print("merge8 shape:", merge8.shape)
    #     conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    #     print("conv8 shape:", conv8.shape)
    #     conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
    #     print("conv8 shape:", conv8.shape)

    #     up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8))
    #     print("up9 shape:", up9.shape)
    #     merge9 = concatenate([conv1, up9], axis=3)
    #     print("merge9 shape:", merge9.shape)
    #     conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    #     print("conv9 shape:", conv9.shape)
    #     conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    #     print("conv9 shape:", conv9.shape)
    #     # conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    #     # print("conv9 shape:", conv9.shape)
    #     conv9 = Conv2D(1, 3, activation='sigmoid', padding='same', kernel_initializer='he_normal')(conv9)
    #     print("conv9 shape:", conv9.shape)
    #     # conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)
    #     # print("conv10 shape:", conv10.shape)
        
	

    #     # model = Model(inputs=input1, outputs=conv10)
    #     model = Model(inputs=input1, outputs=conv9)
	   #  #asdf = dice_coef(y_true, y_pred)
    #     #dice_fn = dicee()


    #     #dloss = dice_coef()
    #     #d     = dice_coef_loss()

    #     lr = 1e-7 #1e-07 or 1e-08
    #     print(lr)
    #     loss = 1
    #     if loss==0:
    #         print("Loss: Binary Crossentropy.")
    #         model.compile(optimizer=Adam(lr=lr),#Adam(lr=1e-5), 
    #                       loss=['binary_crossentropy'], #, dice_coef_loss
    #                       metrics=['accuracy'])#[dice_coef])#,
    #                 #sample_weight_mode='temporal') #loss=dice_coef_loss, 
    #     elif loss==1:
    #         print("Loss: Dice coef loss.")
    #         model.compile(optimizer=Adam(lr=lr), 
    #                       loss=dice_coef_loss,
    #                       metrics=['accuracy'])#[dice_coef])#,

    #     return model

    def get_unet(self):

        inputs = Input((self.img_rows, self.img_cols, 1))
        conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(inputs)
        # conv1 = BatchNormalization(mode=2, axis=3)(conv1)
        conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv1)
        # conv1 = BatchNormalization(mode=2, axis=3)(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(pool1)
        # conv2 = BatchNormalization(mode=2, axis=3)(conv2)
        conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv2)
        # conv2 = BatchNormalization(mode=2, axis=3)(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(pool2)
        # conv3 = BatchNormalization(mode=2, axis=3)(conv3)
        conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv3)
        # conv3 = BatchNormalization(mode=2, axis=3)(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(pool3)
        # conv4 = BatchNormalization(mode=2, axis=3)(conv4)
        conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv4)
        # conv4 = BatchNormalization(mode=2, axis=3)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(pool4)
        # conv5 = BatchNormalization(mode=2, axis=3)(conv5)
        conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv5)
        # conv5 = BatchNormalization(mode=2, axis=3)(conv5)
        # pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)

        # convdeep = Convolution2D(1024, 3, 3, activation='relu', border_mode='same')(pool5)
        # convdeep = Convolution2D(1024, 3, 3, activation='relu', border_mode='same')(convdeep)
        
        # upmid = merge([Convolution2D(512, 2, 2, border_mode='same')(UpSampling2D(size=(2, 2))(convdeep)), conv5], mode='concat', concat_axis=1)
        # convmid = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(upmid)
        # convmid = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(convmid)

        up6 = merge( [Convolution2D(256, 2, 2, activation='relu', border_mode='same')(UpSampling2D(size=(2, 2))(conv5)) ,conv4] , mode='concat', concat_axis=3)
        conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(up6)
        # conv6 = BatchNormalization(mode=2, axis=3)(conv6)
        conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv6)
        # conv6 = BatchNormalization(mode=2, axis=3)(conv6)

        up7 = merge([Convolution2D(128, 2, 2, activation='relu', border_mode='same')(UpSampling2D(size=(2, 2))(conv6)), conv3], mode='concat', concat_axis=3)
        conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(up7)
        # conv7 = BatchNormalization(mode=2, axis=3)(conv7)
        conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv7)
        # conv7 = BatchNormalization(mode=2, axis=3)(conv7)

        up8 = merge([Convolution2D(64, 2, 2, activation='relu', border_mode='same')(UpSampling2D(size=(2, 2))(conv7)), conv2], mode='concat', concat_axis=3)
        conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(up8)
        # conv8 = BatchNormalization(mode=2, axis=3)(conv8)
        conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv8)
        # conv8 = BatchNormalization(mode=2, axis=3)(conv8)

        up9 = merge([Convolution2D(32, 2, 2, activation='relu', border_mode='same')(UpSampling2D(size=(2, 2))(conv8)), conv1], mode='concat', concat_axis=3)
        conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(up9)
        # conv9 = BatchNormalization(mode=2, axis=3)(conv9)
        conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv9)
        # conv9 = BatchNormalization(mode=2, axis=3)(conv9)

        conv10 = Convolution2D(1, 1, 1, activation='sigmoid')(conv9)

        model = Model(input=inputs, output=conv10)

        model.compile(optimizer=Adam(lr=1e-4), loss=dice_coef_loss, metrics=[dice_coef])

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

        # test
        # print(imgs_train.shape)
        # print(imgs_mask_train.shape)
        # import matplotlib.pyplot as plt
        # A = imgs_train[34,:,:,0]
        # B = imgs_mask_train[34,:,:,0]
        # C = imgs_test[34,:,:,0]
        # plt.imshow(A,cmap="gray")
        # plt.show()

        # plt.imshow(B,cmap="gray")
        # plt.show()

        # plt.imshow(C,cmap="gray")
        # plt.show()
        ##

        print("imgs_mask_train Max above 0: " + str(np.max(np.array(imgs_mask_train))))

        print("loading data done")
        
        load = 0
        if load == 0:
            model = self.get_unet()
            print("Created Unet.")
        else:
            model = load_model('unet.hdf5')
            print("Loaded Unet.")
        
        
        model_checkpoint = ModelCheckpoint('weights.{epoch:02d}-{val_loss:.2f}.hdf5', 
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
                batch_size=32, 
                nb_epoch=10, 
                verbose=1, 
                validation_split=0.1, 
                shuffle=True,) 
                # callbacks=[model_checkpoint])#,
                #sample_weight = class_weights)

                    #, class_weight = {0:1, 1:100}

        print('predict test data')
        imgs_mask_test = model.predict(imgs_test, batch_size=1, verbose=1)


        if np.max(np.array(imgs_mask_test))==0.0:
            print(" ==============================")
            print("==== Mask prediction empty. ====")
            print(" ==============================")
        
        np.save('./results/imgs_mask_test.npy', imgs_mask_test)

    def save_img(self):

        print("array to image")
        imgs = np.load('./results/imgs_mask_test.npy')
        print("test result Max: " + str(np.max(np.array(imgs))))
        print("test result Min: " + str(np.min(np.array(imgs))))
        for i in range(imgs.shape[0]):
            img = imgs[i]
            #if i == 30:
                #print("img in unet.py Max: " + str(np.max(np.array(img))))
                #print("img in unet.py Min: " + str(np.min(np.array(img))))
                #print(img.shape)
            if np.isnan(np.sum(img)):
                print("NANI?")

            img = array_to_img(img)
            #print(img.size)
            img.save("./results/%d.tif" % (i))


if __name__ == '__main__':
    myunet = myUnet()
    myunet.train()
    myunet.save_img()
