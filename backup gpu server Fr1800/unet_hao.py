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
from losses import *
from keras.models import load_model

# Global vars for easy access and who cares about programming principles.
_epochs = 5
_which_model = "Keras1_32-512" #ultra. #loadhdf5 #unet_orig #shallow_orig_unet #diff_opt_loss_unet #keras1_unet
_batch_size = 32
_lr = 1e-04
_tensor_in = "N/A"
_test_shape = "N/A"
_features_low = "N/A"
_features_deep = "N/A"
_load = 0
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

    def get_unet(self):
        f = 32 # 32 default. 64 unet.
        global _features_low, _features_deep
        _features_low = f
        _features_deep = f*16

        inputs = Input((self.img_rows, self.img_cols, 1))
        conv1 = Convolution2D(f, 3, 3, activation='relu', border_mode='same')(inputs)
        # conv1 = BatchNormalization(mode=2, axis=3)(conv1)
        conv1 = Convolution2D(f, 3, 3, activation='relu', border_mode='same')(conv1)
        # conv1 = BatchNormalization(mode=2, axis=3)(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Convolution2D(f*2, 3, 3, activation='relu', border_mode='same')(pool1)
        # conv2 = BatchNormalization(mode=2, axis=3)(conv2)
        conv2 = Convolution2D(f*2, 3, 3, activation='relu', border_mode='same')(conv2)
        # conv2 = BatchNormalization(mode=2, axis=3)(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Convolution2D(f*4, 3, 3, activation='relu', border_mode='same')(pool2)
        # conv3 = BatchNormalization(mode=2, axis=3)(conv3)
        conv3 = Convolution2D(f*4, 3, 3, activation='relu', border_mode='same')(conv3)
        # conv3 = BatchNormalization(mode=2, axis=3)(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Convolution2D(f*8, 3, 3, activation='relu', border_mode='same')(pool3)
        # conv4 = BatchNormalization(mode=2, axis=3)(conv4)
        conv4 = Convolution2D(f*8, 3, 3, activation='relu', border_mode='same')(conv4)
        # conv4 = BatchNormalization(mode=2, axis=3)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = Convolution2D(f*16, 3, 3, activation='relu', border_mode='same')(pool4)
        # conv5 = BatchNormalization(mode=2, axis=3)(conv5)
        conv5 = Convolution2D(f*16, 3, 3, activation='relu', border_mode='same')(conv5)
        # conv5 = BatchNormalization(mode=2, axis=3)(conv5)
        # pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)

        # convdeep = Convolution2D(1024, 3, 3, activation='relu', border_mode='same')(pool5)
        # convdeep = Convolution2D(1024, 3, 3, activation='relu', border_mode='same')(convdeep)
        
        # upmid = merge([Convolution2D(512, 2, 2, border_mode='same')(UpSampling2D(size=(2, 2))(convdeep)), conv5], mode='concat', concat_axis=1)
        # convmid = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(upmid)
        # convmid = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(convmid)

        up6 = merge([Convolution2D(f*8, 2, 2, activation='relu', border_mode='same')(UpSampling2D(size=(2, 2))(conv5)), conv4], mode='concat', concat_axis=3)
        conv6 = Convolution2D(f*8, 3, 3, activation='relu', border_mode='same')(up6)
        # conv6 = BatchNormalization(mode=2, axis=3)(conv6)
        conv6 = Convolution2D(f*8, 3, 3, activation='relu', border_mode='same')(conv6)
        # conv6 = BatchNormalization(mode=2, axis=3)(conv6)

        up7 = merge([Convolution2D(f*4, 2, 2, activation='relu', border_mode='same')(UpSampling2D(size=(2, 2))(conv6)), conv3], mode='concat', concat_axis=3)
        conv7 = Convolution2D(f*4, 3, 3, activation='relu', border_mode='same')(up7)
        # conv7 = BatchNormalization(mode=2, axis=3)(conv7)
        conv7 = Convolution2D(f*4, 3, 3, activation='relu', border_mode='same')(conv7)
        # conv7 = BatchNormalization(mode=2, axis=3)(conv7)

        up8 = merge([Convolution2D(f*2, 2, 2, activation='relu', border_mode='same')(UpSampling2D(size=(2, 2))(conv7)), conv2], mode='concat', concat_axis=3)
        conv8 = Convolution2D(f*2, 3, 3, activation='relu', border_mode='same')(up8)
        # conv8 = BatchNormalization(mode=2, axis=3)(conv8)
        conv8 = Convolution2D(f*2, 3, 3, activation='relu', border_mode='same')(conv8)
        # conv8 = BatchNormalization(mode=2, axis=3)(conv8)

        up9 = merge([Convolution2D(f, 2, 2, activation='relu', border_mode='same')(UpSampling2D(size=(2, 2))(conv8)), conv1], mode='concat', concat_axis=3)
        conv9 = Convolution2D(f, 3, 3, activation='relu', border_mode='same')(up9)
        # conv9 = BatchNormalization(mode=2, axis=3)(conv9)
        conv9 = Convolution2D(f, 3, 3, activation='relu', border_mode='same')(conv9)
        # conv9 = BatchNormalization(mode=2, axis=3)(conv9)

        conv10 = Convolution2D(1, 1, 1, activation='sigmoid')(conv9)

        model = Model(input=inputs, output=conv10)
        lr = _lr
        model.compile(optimizer=Adam(lr=lr), loss=dice_coef_loss, metrics=[dice_coef])
        print("Model Specs: Lr: " + str(lr) + " Feature highest/lowest: " + str(f) + "/" + str(f*16) + " Loss: dice_coef_loss") 
        return model



    def train(self):

        print("loading data")
        imgs_train, imgs_mask_train, imgs_test = self.load_data()
        global _tensor_in, _test_shape
        _tensor_in  = imgs_train.shape
        _test_shape = imgs_test.shape

        # test
        print(imgs_train.shape)
        print(imgs_mask_train.shape)
        #import matplotlib.pyplot as plt
        #A = imgs_train[34,:,:,0]
        #B = imgs_mask_train[34,:,:,0]
        #C = imgs_test[34,:,:,0]
        #plt.imshow(A,cmap="gray")
        #plt.show()

        #plt.imshow(B,cmap="gray")
        #plt.show()

        #plt.imshow(C,cmap="gray")
        #plt.show()
        ##

        print("imgs_mask_train Max above 0: " + str(np.max(np.array(imgs_mask_train))))

        print("loading data done")
        
        load = _load
        if load == 0:
            model = self.get_unet()
            print("Created Unet.")
        else:
            model = load_model('unet.hdf5', custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef})
            print("Loaded Unet.")
        
        
        model_checkpoint = ModelCheckpoint('unet.hdf5', #'weights.{epoch:02d}-{val_loss:.2f}.hdf5', 
                                            monitor='loss', 
                                            verbose=1, 
                                            save_best_only=False)
        epochs = _epochs
        batch = _batch_size
        print('Fitting model... InputTensorShape: ' + str(imgs_train.shape) + "Epochs: " + str(epochs) + "batchsize: " + str(batch))
        
        # 1436 images
        #class_weights = np.zeros((1436, 102400))
        #class_weights[:, 0] += 0.05
        #class_weights[:, 1] += 0.95

        

        model.fit(imgs_train, 
                imgs_mask_train, 
                batch_size=batch, 
                nb_epoch=epochs, 
                verbose=1, 
                validation_split=0.1, 
                shuffle=True,
                callbacks=[model_checkpoint])#,
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

    def save_model_info(self):
        file = open("./results/model_info.txt","w")
        file.write("epochs: " + str(_epochs) + "\n")
        file.write("which_model: " + str(_which_model) + "\n")
        file.write("_batch_size: " + str(_batch_size) + "\n")
        file.write("_lr: " + str(_lr) + "\n")
        file.write("_tensor_in: " + str(_tensor_in) + "\n")
        file.write("_test_shape: " + str(_test_shape) + "\n")
        file.write("_features_low: " + str(_features_low) + "\n")
        file.write("_features_deep: " + str(_features_deep) + "\n")
        file.close()
        print("Wrote model info file.")

if __name__ == '__main__':
    myunet = myUnet()
    myunet.train()
    myunet.save_img()
    myunet.save_model_info()
