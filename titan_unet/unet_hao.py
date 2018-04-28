import os, errno
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
import nibabel
import shutil


# Global vars for easy access and who cares about programming principles.
_epochs = 10
_which_model = "Keras1_32-512" #ultra. #loadhdf5 #unet_orig #shallow_orig_unet #diff_opt_loss_unet #keras1_unet
_batch_size = 8
_lr = 1e-04
_tensor_in = "N/A"
_test_shape = "N/A"
_features_low = "N/A"
_features_deep = "N/A"
_load = 0 
#0 is new, else is load

_nii_test_img = None
_nii_test_mask = None

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
                #sample_weight = class_weights) # No idea how to make these work currently.
                #, class_weight = {0:1, 1:100}

        #print('predict test data')
        #imgs_mask_test = model.predict(imgs_test, batch_size=1, verbose=1)
        #np.save('./results/imgs_mask_test.npy', imgs_mask_test)
        return model

    ###
    def pred_save_test_datas(self, model):
        model.load_weights('unet.hdf5')
        print('Predicting multiple test data...')

        # Must fetch data from './npydata/test_npy/'

        # Make sure folder exists where we want to save.
        try:
            print("Making folder: ./results/mask_pred_npys/")
            os.makedirs('./results/mask_pred_npys/')
        except OSError as e:
            #print("... folder already exists or other error.")
            if e.errno != errno.EEXIST:
                raise

        # Make sure folder exists where we want to save.
        try:
            print("Making folder: ./npydata/test_npy/")
            os.makedirs('./npydata/test_npy/')
        except OSError as e:
            #print("... folder already exists or other error.")
            if e.errno != errno.EEXIST:
                raise

        # Set where the test data is saved.
        parent_folder = './npydata/test_npy/'
        test_files = os.listdir(parent_folder)
        print(test_files) #['NR_Diff_45.npy', 'NR_Diff_34.npy', etc

        # Get every mri scan to be tested. (it's .npy files now)
        print("3445")
        for test_file in test_files:
            print("test_file" + str(test_file)) #NR_Diff_59.npy
            # Load a file at a time.
            img_test = np.load(parent_folder + test_file)
            #print(img_test.shape) # (39, 256, 256, 1)
            
            # Set to float32.
            img_test = img_test.astype('float32')

            # Do prediction.
            print("Predicting file: " + str(test_file))
            test_mask_prediction = model.predict(img_test, batch_size=1, verbose=1)

            # Save predicted mask.
            np.save('./results/mask_pred_npys/' + test_file, test_mask_prediction) #TODO
    ###

    ###
    def save_all_predictions_as_nii(self):
        print("Saving predictions to /results/ folder.")
        
        # All the prediction masks .npy files are in: './results/mask_pred_npys/' e.g. NR_Diff_02.npy

        # Make sure folder exists where we want to save.
        try:
            print("Making folder: ./results/images/")
            os.makedirs('./results/images/')
        except OSError as e:
            #print("... folder already exists or other error.")
            if e.errno != errno.EEXIST:
                raise

        # Set where to load the prediction .npy files from.
        parent_folder = './results/mask_pred_npys/'
        pred_files = os.listdir(parent_folder)

        # Go through all files which are a stack each.
        for pred_file in pred_files:
            #print(pred_file) # NR_Diff_45.npy
            # Load the predicted mask .npy file. One STACK at a time.
            pred_mask = np.load(parent_folder + pred_file) # shape: 41, 256, 256

            #nr_diff_xx = os.rename(pred_file, pred_file.replace('.npy', '')) # for existing folder, not variable
            nr_diff_xx = pred_file.replace('.npy', '')

            # Make sure folder exists where we want to save.
            try:
                print("Making folder: ./results/images/" + nr_diff_xx + "/")
                os.makedirs('./results/images/' + nr_diff_xx + '/')
            except OSError as e:
                #print("... folder already exists or other error.")
                if e.errno != errno.EEXIST:
                    raise
            
            # Loop through the stack.
            for i in range(pred_mask.shape[0]):
                # Get 1 img from stack at a time.
                img = pred_mask[i]
                # Threshhold to 0 and 1.
                img[img >= 0.5] = 1
                img[img < 0.5] = 0
                # Turn into image 2D.
                img = array_to_img(img)
                img.save("./results/images/" + nr_diff_xx + "/%d.tif" % (i))

            # Now turn into .nii, resize (pad to 276*320) and add correct header and affine.
            # Original .nii files are in './TumourData/test_nii/NR_Diff_XX/' and here the nii.gz
            
            #pred_file  is NR_Diff_45.npy   # This is a stack.
            #nr_diff_xx is NR_Diff_45

            # Let's load original to steal .header and .affine from the .nii file. They are in ./TumourData/test_nii/NR_Diff_XX/FLAIR_mask.nii.gz
            mask_nii = nibabel.load('./TumourData/test_nii/' + nr_diff_xx + '/' + 'FLAIR_mask.nii.gz')
            hdr = mask_nii.header
            aff = mask_nii.affine

            # Pad the numpyarray pred_mask back into xx*276*320 (currently xx*256*256). 
            print("pred_mask.shape before slice: " + str(pred_mask.shape))
            pred_mask = pred_mask[:,:,:,0]
            print("pred_mask.shape before pad: " + str(pred_mask.shape))
            pred_mask = np.pad(pred_mask, ((0,0),(10,10),(32,32)), 'constant')
            print("pred_mask.shape after pad: " + str(pred_mask.shape))


            # Transpose numpy array from channel first into channel last. ###np.transpose(aa, (1,2,0)) # from (41, 256, 256) to (256, 256, 41)
            pred_mask = np.transpose(pred_mask, (1,2,0))

            # Make folder?
            try:
                print("Making folder: ./results/nii_masks/")
                os.makedirs('./results/nii_masks/')
            except OSError as e:
                #print("... folder already exists or other error.")
                if e.errno != errno.EEXIST:
                    raise
            
            # Threshold to 1 and 0.
            pred_mask[pred_mask >= 0.5] = 1
            pred_mask[pred_mask < 0.5] = 0
            # Save as .nii
            niimask_new = nibabel.Nifti1Image(pred_mask, affine=aff, header=hdr)
            nibabel.save(niimask_new, './results/nii_masks/' + nr_diff_xx + '_Unet.nii.gz') 

    ###

    def save_img(self):

        print("Saving predictions to /results/ folder.")
        mask_preds = np.load('./results/imgs_mask_test.npy')

        #mask_preds[mask_preds >= 0.5] = 1
        #mask_preds[mask_preds < 0.5] = 0

        print("test result Max: " + str(np.max(np.array(mask_preds))))
        print("test result Min: " + str(np.min(np.array(mask_preds))))
        for i in range(mask_preds.shape[0]):
            img = mask_preds[i]
            img[img >= 0.5] = 1
            img[img < 0.5] = 0
    	    #if i == 30:
                #print("img in unet.py Max: " + str(np.max(np.array(img))))
                #print("img in unet.py Min: " + str(np.min(np.array(img))))
                #print(img.shape)
            if np.isnan(np.sum(img)):
                print("NANI?")

            img = array_to_img(img)
            #img[img >= 0.5] = 1
	    #img[img < 0.5] = 0 doesnt work
	    #print(img.size)
            img.save("./results/%d.tif" % (i))

        # Turn into nifti1 and resize to 256, update header info and save.
        # ./TumourData/test_nii/ [only 1 folder here]
        parent_folder = './TumourData/test_nii/'
        patient_folders = os.listdir(parent_folder)

        for patient_folder in patient_folders: #this only runs once, as only 1 folder there at a time.
            # patient_folder is NR_Diff_XX so can use to name output.
            image_nii   = nibabel.load(parent_folder + patient_folder + '/' + 'flair_noskull.nii.gz')
            mask_nii    = nibabel.load(parent_folder + patient_folder + '/' + 'FLAIR_mask.nii.gz')
            
            print(mask_preds.shape) #41, 256, 256
            mask_preds = mask_preds[:,:,:,0]
            print(mask_preds.shape)
            #mask_preds2 = np.transpose(mask_preds, (2,0,1))
            mask_preds2 = np.transpose(mask_preds, (1,2,0)) # swapping axes does weird things.
            #mask_preds2 = np.swapaxes(mask_preds, 0, 2)
            #mask_preds2 = np.rollaxis(mask_preds, 2,0)
            print("mask_preds.shape: " + str(mask_preds.shape) + " must equal order mask_nii.shape: " + str(mask_nii.shape))
            #print("mask_preds1.shape: " + str(mask_preds1.shape) + " must equal order mask_nii.shape: " + str(mask_nii.shape))
            print("mask_preds2.shape: " + str(mask_preds2.shape) + " must equal order mask_nii.shape: " + str(mask_nii.shape))

            # Adjust headers to 256*256 dimension.
            image_nii_header = image_nii.header
            image_nii_header['dim'][2] = 256
            image_nii_header['dim'][1] = 256
            mask_nii_header = mask_nii.header
            mask_nii_header['dim'][2] = 256
            mask_nii_header['dim'][1] = 256

            image_numpy = image_nii.get_data()
            
            if 1==1: #resize image_numpy to 256
                a1, b1, c1 = image_numpy.shape
                print("=== === Img Shape before crop: " + str(image_numpy.shape))
                half_excess_a1 = int( (a1 - 256) / 2 )
                half_excess_b1 = int( (b1 - 256) / 2 )
                image_numpy = image_numpy[0+half_excess_a1:a1-half_excess_a1,
                                          0+half_excess_b1:b1-half_excess_b1,
                                          :] #not a smiley.
                print("=== === Img Shape after crop:  " + str(image_numpy.shape))

            print("image_numpy.shape: " + str(image_numpy.shape) + " must equal order image_nii.shape: " + str(image_nii.shape))

            #Must create "patient folder".
            try:
                print("Trying to make folder...")
                os.makedirs('./results/' + patient_folder)
            except OSError as e:
                print("... no success making folder.")
                if e.errno != errno.EEXIST:
                    raise

            # Save to .nii
            nii_img_new = nibabel.Nifti1Image(image_numpy, affine=image_nii.affine, header=image_nii_header)
            nibabel.save(nii_img_new, './results/' + patient_folder + '/flair_noskull_256_Unet.nii.gz')

            mask_preds2[mask_preds2 >= 0.5] = 1
            mask_preds2[mask_preds2 < 0.5] = 0
            niimask_new1 = nibabel.Nifti1Image(mask_preds2, affine=image_nii.affine, header=image_nii_header)
            nibabel.save(niimask_new1, './results/' + patient_folder + '/FLAIR_mask_256_Unet000.nii.gz') 
            niimask_new2 = nibabel.Nifti1Image(mask_preds2, affine=image_nii.affine, header=mask_nii_header)
            nibabel.save(niimask_new2, './results/' + patient_folder + '/FLAIR_mask_256_Unet111.nii.gz')
            niimask_new3 = nibabel.Nifti1Image(mask_preds2, affine=mask_nii.affine, header=mask_nii_header)
            nibabel.save(niimask_new3, './results/' + patient_folder + '/FLAIR_mask_256_Unet222.nii.gz') 

            


    ###
    def prep_result_folder(self):
        shutil.rmtree('./results')
        # Remake ./results
        # Make folder?
        try:
            print("Making folder: ./results/")
            os.makedirs('./results/')
        except OSError as e:
            #print("... folder already exists or other error.")
            if e.errno != errno.EEXIST:
                raise
    ###

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
    model = myunet.train()
    #myunet.prep_result_folder() # deletes existing results folder and remakes it empty
    myunet.pred_save_test_datas(model) # predicts multiple test datas and saves them as npy
    myunet.save_all_predictions_as_nii()
    #myunet.save_img() # Saves only 1 npy file.
    myunet.save_model_info()
    print("Last Thing unet_hao.py prints. The End.")
