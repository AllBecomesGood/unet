from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
import os, errno
import glob
from numpy import mean, std
import nibabel
from scipy.ndimage import rotate
import shutil

_augmentation = True
_flipping = True
_rota15 = True
_flip_rota = False #seems to worsen results
_folder_test_data = "N/A"


class dataProcess(object):

	def __init__(self, out_rows, out_cols,
				train_path = "./deform/train_nii",
				label_path = "./deform/labels_nii",
				test_path = "./deform/test_nii",
				npy_path = "./npydata",
				img_type = "tif"):

		self.out_rows = out_rows
		self.out_cols = out_cols
		self.train_path = train_path
		self.label_path = label_path
		self.img_type = img_type
		self.test_path = test_path
		self.npy_path = npy_path


	def create_train_data_nii(self):
		"""
		Load training images and mask(label) images.
		Turn them into numpy arrays, apply zero mean and unit standard deviation
		 and save them in a list.
		Save them into .npy files.
		"""
		print('-'*30)
		print('Loading training images and masks...')

		augmentation = _augmentation
		flipping = _flipping
		rota15 = _rota15
		flip_rota = _flip_rota

		i = 0
		# Load the images.
		parent_folder = './TumourData/Kurtosis_Gliomas_nii/'
		patient_folders = os.listdir(parent_folder)

		numOfImagesTotalBase = 0
		for patient_folder in patient_folders:
			image_nii = nibabel.load(parent_folder + patient_folder + '/' + 'flair_noskull.nii.gz')
			_, _, num_of_slices = image_nii.shape
			numOfImagesTotalBase += num_of_slices
		#print("num of slices total: " + str(numOfImagesTotal))
		
		numOfImagesTotal = numOfImagesTotalBase # 1 set is minimum. Adding space as needed per augmentation.
		print("Base amount of un-augmented training data is: " + str(numOfImagesTotalBase))
		if augmentation == True:
			if rota15 == True:
				numOfImagesTotal += numOfImagesTotalBase * 2 # 1 set +15 degrees and 1 set -15 degrees.
				print("Rotation increased amount of training data to: " + str(numOfImagesTotal))
			if flipping == True:
				numOfImagesTotal += numOfImagesTotalBase * 3 # 1*updown, 1*leftright, and then combination of both so 1 more set. 3 overall.
				print("Flipping increased namount of training data to: " + str(numOfImagesTotal))
			if flip_rota == True:
				numOfImagesTotal += numOfImagesTotalBase * 2 # LR flip pos and neg rota -> 2 more.
				print("flip_rota increased amount of training data to: " + str(numOfImagesTotal))

		# Create empty placeholder numpy array. (dim here: 30,512,512,1)
		imgdatas = np.ndarray((numOfImagesTotal,self.out_rows,self.out_cols, 1), dtype=np.float32) 
		imglabels = np.ndarray((numOfImagesTotal,self.out_rows,self.out_cols, 1), dtype=np.float32)#TODO: float32/uint8
		
		if i == 0:
			print("# of folders/patients should be 36 (37-1test): " + str(len(patient_folders)))
		ff = 10 # set 0 to run debug. set above 10 to not run.
		# loops thru imgs
		for patient_folder in patient_folders:
			#midname = imgname[imgname.rindex("/")+1:]
			image_nii        = nibabel.load(parent_folder + patient_folder + '/' + 'flair_noskull.nii.gz')
			image_nii_mask   = nibabel.load(parent_folder + patient_folder + '/' + 'FLAIR_mask.nii.gz')
			image_numpy      = image_nii.get_data()
			image_numpy_mask = image_nii_mask.get_data()

			# Crop / pad / whatever must be done before zero-mean, as it'd affect it otherwise.
			# Crop img and mask to 256, keep centre portion of img, so cut off outside area.
			a1, b1, c1 = image_numpy.shape
			half_excess_a1 = int( (a1 - 256) / 2 )
			half_excess_b1 = int( (b1 - 256) / 2 )
			image_numpy = image_numpy[0+half_excess_a1:a1-half_excess_a1,
									  0+half_excess_b1:b1-half_excess_b1,
									  :] #not a smiley.

			a2, b2, c2 = image_numpy_mask.shape
			half_excess_a2 = int( (a2 - 256) / 2 )
			half_excess_b2 = int( (b2 - 256) / 2 )
			image_numpy_mask = image_numpy_mask[0+half_excess_a2:a2-half_excess_a2,
									  0+half_excess_b2:b2-half_excess_b2,
									  :]
			
			if c1 != c2:
				print("Imgs and mask numbers not equal."*5)
			
			if rota15 == True:
				#from scipy.ndimage import rotate
				image_numpy_rotated_positive = rotate(image_numpy, 15, axes=(2, 1), reshape=False)
				mask_numpy_rotated_positive  = rotate(image_numpy_mask, 15, axes=(2, 1), reshape=False)

				image_numpy_rotated_minus = rotate(image_numpy, -15, axes=(2, 1), reshape=False)
				mask_numpy_rotated_minus  = rotate(image_numpy_mask, -15, axes=(2, 1), reshape=False)
				
				
				if ff == 0:
					print("Rota debug.")
					print("image_numpy.shape: " + str(image_numpy.shape))
					print("Nan? " + str(np.isnan(np.sum(image_numpy))))
					print("image_numpy_mask.shape: " + str(image_numpy_mask.shape))
					print("Nan? " + str(np.isnan(np.sum(image_numpy_mask))))
					print("image_numpy_rotated_positive.shape: " + str(image_numpy_rotated_positive.shape))
					print("Nan? " + str(np.isnan(np.sum(image_numpy_rotated_positive))))
					print("mask_numpy_rotated_positive.shape: " + str(mask_numpy_rotated_positive.shape))
					print("Nan? " + str(np.isnan(np.sum(mask_numpy_rotated_positive))))
					print("image_numpy_rotated_minus.shape: " + str(image_numpy_rotated_minus.shape))
					print("Nan? " + str(np.isnan(np.sum(image_numpy_rotated_minus))))
					print("mask_numpy_rotated_minus.shape: " + str(mask_numpy_rotated_minus.shape))
					print("Nan? " + str(np.isnan(np.sum(mask_numpy_rotated_minus))))
					print("image_numpy_rotated_positive Max: " + str(np.max(np.array(image_numpy_rotated_positive))))
					print("Nan? " + str(np.isnan(np.sum(image_numpy_rotated_positive))))
					print("image_numpy_rotated_positive Min: " + str(np.min(np.array(image_numpy_rotated_positive))))
					print("Nan? " + str(np.isnan(np.sum(image_numpy_rotated_positive))))
					print("Rota debug end.")
					ff = ff + 1



			#zero mean	
			image_numpy = (image_numpy - mean(image_numpy)) / std(image_numpy)
			if rota15 == True:
				# Must normalise after rotation, as 0's are introduced which would affect it, since after zero-mean the new min would be negative.
				# Image and Mask.
				image_numpy_rotated_positive = (image_numpy_rotated_positive - mean(image_numpy_rotated_positive)) / std(image_numpy_rotated_positive)
				#mask_numpy_rotated_positive  = (mask_numpy_rotated_positive  - mean(mask_numpy_rotated_positive))  / std(mask_numpy_rotated_positive)

				image_numpy_rotated_minus = (image_numpy_rotated_minus    - mean(image_numpy_rotated_minus)) / std(image_numpy_rotated_minus)
				#mask_numpy_rotated_minus  = (mask_numpy_rotated_minus     - mean(mask_numpy_rotated_minus))  / std(mask_numpy_rotated_minus)
				if ff == 1:
					print("Rota 2 debug.")
					print("image_numpy.shape: " + str(image_numpy.shape))
					print("image_numpy_mask.shape: " + str(image_numpy_mask.shape))
					print("image_numpy_rotated_positive.shape: " + str(image_numpy_rotated_positive.shape))
					print("mask_numpy_rotated_positive.shape: " + str(mask_numpy_rotated_positive.shape))
					print("image_numpy_rotated_minus.shape: " + str(image_numpy_rotated_minus.shape))
					print("mask_numpy_rotated_minus.shape: " + str(mask_numpy_rotated_minus.shape))
					print("image_numpy_rotated_positive Max: " + str(np.max(np.array(image_numpy_rotated_positive))))
					print("image_numpy_rotated_positive Min: " + str(np.min(np.array(image_numpy_rotated_positive))))
					print("Rota 2 debug end.")
					ff = ff + 1

			#print("image_numpy in data.py Max: " + str(np.max(np.array(image_numpy))))
			#print("image_numpy in data.py Min (negative): " + str(np.min(np.array(image_numpy))))
				


			for x in range(0, c1):
				# zero mean and unit standard deviation
				#image_numpy[:,:,x] = (image_numpy[:,:,x] - mean(image_numpy[:,:,x])) / std(image_numpy[:,:,x])
				
				if augmentation == False:
					# We lose a dimension when we take out one slice ie [:,:,x]. 
					# Return to dim1xdim2x1 shape via img_to_array()
					img_train = img_to_array(image_numpy[:,:,x])
					mask = img_to_array(image_numpy_mask[:,:,x])
					
					imgdatas[i]  = img_train
					imglabels[i] = mask
					i += 1
				# elif augment_type == 1:
				# 	# We lose a dimension when we take out one slice. Return to dim1xdim2x1 shape.
				# 	img_train = img_to_array(image_numpy[:,:,x])
				# 	img_train_rota1 = img_to_array( np.rot90(image_numpy[:,:,x]) )
				# 	img_train_rota2 = img_to_array( np.rot90(image_numpy[:,:,x], 2) )
				# 	img_train_rota3 = img_to_array( np.rot90(image_numpy[:,:,x], 3) )

				# 	mask = img_to_array(image_numpy_mask[:,:,x])
				# 	mask_rota1 = img_to_array( np.rot90(image_numpy_mask[:,:,x]) )
				# 	mask_rota2 = img_to_array( np.rot90(image_numpy_mask[:,:,x], 2) )
				# 	mask_rota3 = img_to_array( np.rot90(image_numpy_mask[:,:,x], 3) )

				# 	imgdatas[i]  = img_train
				# 	imglabels[i] = mask
				# 	i += 1
				# 	imgdatas[i]  = img_train_rota1
				# 	imglabels[i] = mask_rota1
				# 	i += 1
				# 	imgdatas[i]  = img_train_rota2
				# 	imglabels[i] = mask_rota2
				# 	i += 1
				# 	imgdatas[i]  = img_train_rota3
				# 	imglabels[i] = mask_rota3
				# 	i += 1
				elif augmentation == True:
					# We lose a dimension when we take out one slice. Return to dim1xdim2x1 shape.
					img_train = img_to_array( image_numpy[:,:,x] )
					if flipping == True:
						img_train_fliplr    = img_to_array( np.fliplr(image_numpy[:,:,x]) ) # left right flip
						img_train_flipud    = img_to_array( np.flipud(image_numpy[:,:,x]) ) # up down flip
						img_train_fliplr_ud = img_to_array( np.flipud(np.fliplr(image_numpy[:,:,x])) ) # flip upDown AND leftRight

					mask 		   = img_to_array( image_numpy_mask[:,:,x] )
					if flipping == True:
						mask_fliplr    = img_to_array( np.fliplr(image_numpy_mask[:,:,x]) ) # left right flip
						mask_flipud    = img_to_array( np.flipud(image_numpy_mask[:,:,x]) ) # up down flip
						mask_fliplr_ud = img_to_array( np.flipud(np.fliplr(image_numpy_mask[:,:,x])) ) # flip upDown AND leftRight

					imgdatas[i]  = img_train
					imglabels[i] = mask
					i += 1
					if flipping == True:
						imgdatas[i]  = img_train_fliplr
						imglabels[i] = mask_fliplr
						i += 1
						imgdatas[i]  = img_train_flipud
						imglabels[i] = mask_flipud
						i += 1
						imgdatas[i]  = img_train_fliplr_ud
						imglabels[i] = mask_fliplr_ud
						i += 1
					if rota15 == True:
						imgdatas[i]  = img_to_array( image_numpy_rotated_positive[:,:,x] )
						imglabels[i] = img_to_array( mask_numpy_rotated_positive[:,:,x] )
						i += 1
						imgdatas[i]  = img_to_array( image_numpy_rotated_minus[:,:,x] )
						imglabels[i] = img_to_array( mask_numpy_rotated_minus[:,:,x] )
						i += 1
					if flip_rota == True: #TODO only this so far
						# Left Right Flip the rotated imgs.
						flip_rota_pos_img  = img_to_array( np.fliplr(image_numpy_rotated_positive[:,:,x]) )
						flip_rota_pos_mask = img_to_array( np.fliplr(mask_numpy_rotated_positive[:,:,x]) )
						flip_rota_minus_img  = img_to_array( np.fliplr(image_numpy_rotated_minus[:,:,x]) )
						flip_rota_minus_mask = img_to_array( np.fliplr(mask_numpy_rotated_minus[:,:,x]) )
						if ff==2:
							print("flip_rota_pos_img.shape: " + str(flip_rota_pos_img.shape))
							print("flip_rota_pos_mask.shape: " + str(flip_rota_pos_mask.shape))
							print("flip_rota_minus_img.shape: " + str(flip_rota_minus_img.shape))
							print("flip_rota_minus_mask.shape: " + str(flip_rota_minus_mask.shape))
							print("flip_rota_pos_img Max positive?: " + str(np.max(np.array(flip_rota_pos_img))))
							print("flip_rota_pos_img Min negative?: " + str(np.min(np.array(flip_rota_pos_img))))
							print("flip_rota_pos_mask Max be 1? (could be all black mask): " + str(np.max(np.array(flip_rota_pos_mask))))
							print("flip_rota_pos_mask Min be 0?: " + str(np.min(np.array(flip_rota_pos_mask))))
							ff = ff + 1
						imgdatas[i]  = flip_rota_pos_img
						imglabels[i] = flip_rota_pos_mask
						i += 1
						imgdatas[i]  = flip_rota_minus_img
						imglabels[i] = flip_rota_minus_mask
						i += 1


				_, _, channel = img_train.shape
				_, _, channel_mask = mask.shape
				if channel != 1:
					print("!!! Wrong Channel! !!! e444")
				if channel_mask != 1:
					print("!!! Wrong Channel! !!! e445")
			

		print('Loaded {0}/{1} images.'.format(i, numOfImagesTotal))

		print("imgdatas.shape 7777" + str(imgdatas.shape))
		np.save(self.npy_path + '/imgs_train.npy', imgdatas)
		np.save(self.npy_path + '/imgs_mask_train.npy', imglabels)

		print("imgdatas.shape: " + str(imgdatas.shape))
		print("imglabels.shape: " + str(imglabels.shape))
		print('Saved train-data and masks to .npy file.')
		print('-'*30)

	def create_test_data_nii(self):
		print('-'*30)
		print("Entering function: create_test_data_nii()")
		print('Creating test images...')

		i = 0
		# Load the images.
		parent_folder = './TumourData/test_nii/'
		patient_folders = os.listdir(parent_folder)

		numOfImagesTotal = 0
		for patient_folder in patient_folders:
			global _folder_test_data
			_folder_test_data = patient_folder
			image_nii = nibabel.load(parent_folder + patient_folder + '/' + 'flair_noskull.nii.gz')
			_, _, num_of_slices = image_nii.shape
			numOfImagesTotal += num_of_slices
		#print("num of slices total: " + str(numOfImagesTotal))
		
		# Create empty placeholder numpy array. (dim here: 30,512,512,1)
		imgdatas = np.ndarray((numOfImagesTotal,self.out_rows,self.out_cols, 1), dtype=np.float32) #MondayNightTODO was ,1 dim
		
		if i == 0:
			print("# of folders/patients should be 2 (37-35train): " + str(len(patient_folders)))
		# Loop thru folders which contain an MRI scan each.
		for patient_folder in patient_folders:
			image_nii   = nibabel.load(parent_folder + patient_folder + '/' + 'flair_noskull.nii.gz')
			image_numpy = image_nii.get_data()
			
			a1, b1, c1 = image_numpy.shape
			print("=== === Img Shape before crop: " + str(image_numpy.shape))
			half_excess_a1 = int( (a1 - 256) / 2 )
			half_excess_b1 = int( (b1 - 256) / 2 )
			image_numpy = image_numpy[0+half_excess_a1:a1-half_excess_a1,
									  0+half_excess_b1:b1-half_excess_b1,
									  :] #not a smiley.
			print("=== === Img Shape after crop:  " + str(image_numpy.shape))
			
			# Zero-mean and unit standard deviation.
			image_numpy = (image_numpy - mean(image_numpy)) / std(image_numpy)
				


			for x in range(0, c1):
				# zero mean and unit standard deviation
				#image_numpy[:,:,x] = (image_numpy[:,:,x] - mean(image_numpy[:,:,x])) / std(image_numpy[:,:,x])
				
				imgdatas[i]  = img_to_array(image_numpy[:,:,x]) # When slicing with [:,:,x] we lose the third dimension.
				i += 1

		print('loading done')
		np.save(self.npy_path + '/imgs_test.npy', imgdatas)
		print('Saving to imgs_test.npy files done.')

		print("Finished function: def create_test_data_nii(self):")
		print('-'*30)
	

	######
	def create_test_data_nii_many(self):
		"""
		Loads .nii from './Tum18_19/test_nii/' folder and saves them as .npy files in './npydata/test_npy/' and names them NR_Diff_XX.npy
		"""

		parent_folder = './TumourData/test_nii/'
		patient_folders = os.listdir(parent_folder)
		print("patient_folders should be 2 (59 and 60): " + str(patient_folders)) # ['NR_Diff_34', 'NR_Diff_60', ...

		for patient_folder in patient_folders:
		    print("333patient_folder: " + str(patient_folder)) # NR_Diff_34 etc
		    
		    # Now we create a numpy testfile for every single one individually?
		    image_nii = nibabel.load(parent_folder + patient_folder + '/' + 'flair_noskull.nii.gz')
		    #print(image_nii.shape) # (276, 320, 41)   not always 41, but same order
		    
		    # Preallocate numpy array where mask will be held in.
		    _, _, slices = image_nii.shape
		    imgdatas = np.ndarray((slices, 256,256, 1), dtype=np.float32)
		    
		    # Turn into numpy data format.
		    image_numpy = image_nii.get_data()
		    #print(image_numpy.shape) # (276, 320, 41)
		    
		    # Crop into 256*256.
		    a1, b1, c1 = image_numpy.shape
		    #print("=== === Img Shape before crop: " + str(image_numpy.shape)) #(276, 320, 41)
		    half_excess_a1 = int( (a1 - 256) / 2 )
		    half_excess_b1 = int( (b1 - 256) / 2 )
		    image_numpy = image_numpy[0+half_excess_a1:a1-half_excess_a1,
		                              0+half_excess_b1:b1-half_excess_b1,
		                              :] #not a smiley.
		    #print("=== === Img Shape after crop:  " + str(image_numpy.shape)) #(256, 256, 41)
		    
		    # Zero-mean and unit standard deviation.
		    image_numpy = (image_numpy - mean(image_numpy)) / std(image_numpy)
		    
		    # Add entire stack into list.
		    for x in range(0, c1):
		        #When slicing with [:,:,x] we lose the third dimension.
		        imgdatas[x] = img_to_array(image_numpy[:,:,x])
		        
		    print(imgdatas.shape) # (41, 256, 256, 1)
		    
		    # Create folder if not already exists.
		    try:
		        print("Making folder...")
		        os.makedirs('./npydata/test_npy/')
		    except OSError as e:
		        #print("... folder already exists or other error.")
		        if e.errno != errno.EEXIST:
		            raise
		    
		    # Save to .np file.
		    print("WHAT IS HAPPENING")
		    print('./npydata/test_npy/' + str(patient_folder) + '.npy')
		    np.save('./npydata/test_npy/' + patient_folder + '.npy', imgdatas)
	######


	def load_train_data(self):
		print('-'*30)
		print('Loading training images and masks...')
		print('-'*30)
		imgs_train = np.load(self.npy_path+"/imgs_train.npy")
		imgs_mask_train = np.load(self.npy_path+"/imgs_mask_train.npy")

		if np.min(np.array(imgs_train)) >= 0:
			print("negative min?")
			print("imgs_train in data.py Max: " + str(np.max(np.array(imgs_train))))
			print("imgs_train in data.py Min: " + str(np.min(np.array(imgs_train))))

		imgs_train = imgs_train.astype('float32')
		imgs_mask_train = imgs_mask_train.astype('float32')
		# imgs are not scaled to 255 and masks are already 0 or 1
		#imgs_train /= 255
		if np.min(np.array(imgs_train)) >= 0:
			print("negative min still?")
			print("imgs_train in data.py Max: " + str(np.max(np.array(imgs_train))))
			print("imgs_train in data.py Min: " + str(np.min(np.array(imgs_train))))
		
		print("imgs_mask_train in data.py Max should be 1: " + str(np.max(np.array(imgs_mask_train))))
		print("imgs_mask_train in data.py Min should be 0: " + str(np.min(np.array(imgs_mask_train))))
		#imgs_mask_train /= 255
		#imgs_mask_train[imgs_mask_train > 0.5] = 1
		#imgs_mask_train[imgs_mask_train <= 0.5] = 0
		return imgs_train, imgs_mask_train

	def load_test_data(self):
		print('-'*30)
		print('Loading test images...')

		imgs_test = np.load(self.npy_path+"/imgs_test.npy")
		imgs_test = imgs_test.astype('float32')
		#imgs_test /= 255
		print("Test img dimensions: " + str(imgs_test.ndim))
		print("test data Max: " + str(np.max(np.array(imgs_test))))
		print("test data Min: " + str(np.min(np.array(imgs_test))))

		return imgs_test

	def save_model_info_data(self, dim1, dim2):
		file = open("./results/data_model_info.txt","w")
		file.write("Test data Folder: " + str(_folder_test_data) + "\n")
		file.write("img dimensions: " + str(dim1) + "x" + str(dim2) + "\n")
		file.write("_augmentation: " + str(_augmentation) + "\n")
		file.write("_flipping: " + str(_flipping) + "\n")
		file.write("_rota15: " + str(_rota15) + "\n")
		file.write("_flip_rota: " + str(_flip_rota) + "\n")
		file.close()
		print("Wrote data model info file.")

	###
	def prep_result_folder(self):
		try:
		    print("Making folder: ./results/")
		    os.makedirs('./results/')
		except OSError as e:
		    #print("... folder already exists or other error.")
		    if e.errno != errno.EEXIST:
		        raise

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

		
		try:
		    print("Making folder: ./npydata/test_npy/")
		    os.makedirs('./npydata/test_npy/')
		except OSError as e:
		    #print("... folder already exists or other error.")
		    if e.errno != errno.EEXIST:
		        raise

		shutil.rmtree('./npydata/test_npy/')

		# Remake ./results
		# Make folder?
		try:
		    print("Making folder: ./npydata/test_npy//")
		    os.makedirs('./npydata/test_npy/')
		except OSError as e:
		    #print("... folder already exists or other error.")
		    if e.errno != errno.EEXIST:
		        raise
    ###
# End dataProcess class.


if __name__ == "__main__":

	print("\n")
	print('='*30)
	dim1 = 256
	dim2 = 256
	mydata = dataProcess(dim1,dim2)
	mydata.prep_result_folder()
	print("Calling function .create_train_data()")
	mydata.create_train_data_nii()
	#print("Calling function .create_test_data()")
	#mydata.create_test_data_nii()

	print("PLEASE WORK")
	mydata.create_test_data_nii_many()

	

	mydata.save_model_info_data(dim1, dim2)
	print('='*30)
	print("\n")
	#imgs_train,imgs_mask_train = mydata.load_train_data()
	#print imgs_train.shape,imgs_mask_train.shape






"""Old unused and unloved code and things.


# No longer padded, as cropped instead.
			# Pad from 276x320 to 320x320 so Conv layers divide evenly 4 times.
			#image_numpy      = np.pad(image_numpy, ((22,22),(0,0),(0,0)), 'constant')
			#image_numpy_mask = np.pad(image_numpy_mask, ((22,22),(0,0),(0,0)), 'constant')
			#print("new: shape img: " + str(image_numpy.shape) + " shape mask: " + str(image_numpy_mask.shape))






"""

