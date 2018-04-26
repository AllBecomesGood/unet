from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
import os
import glob
#from sklearn.preprocessing import StandardScaler 
from numpy import mean, std
import nibabel
#import cv2
#from libtiff import TIFF

'''
class myAugmentation(object):

	"""
	A class used to augment image
	Firstly, read train image and label seperately, and then merge them together for the next process
	Secondly, use keras preprocessing to augmentate image
	Finally, seperate augmentated image apart into train image and label
	"""

	def __init__(self, train_path="train", label_path="label", merge_path="merge", aug_merge_path="aug_merge", aug_train_path="aug_train", aug_label_path="aug_label", img_type="tif"):

		"""
		Using glob to get all .img_type from path
		"""

		self.train_imgs = glob.glob(train_path+"/*."+img_type)
		self.label_imgs = glob.glob(label_path+"/*."+img_type)
		self.train_path = train_path
		self.label_path = label_path
		self.merge_path = merge_path
		self.img_type = img_type
		self.aug_merge_path = aug_merge_path
		self.aug_train_path = aug_train_path
		self.aug_label_path = aug_label_path
		self.slices = len(self.train_imgs)
		self.datagen = ImageDataGenerator(
							        rotation_range=0.2,
							        width_shift_range=0.05,
							        height_shift_range=0.05,
							        shear_range=0.05,
							        zoom_range=0.05,
							        horizontal_flip=True,
							        fill_mode='nearest')

	def Augmentation(self):

		"""
		Start augmentation.....
		"""
		trains = self.train_imgs
		labels = self.label_imgs
		path_train = self.train_path
		path_label = self.label_path
		path_merge = self.merge_path
		imgtype = self.img_type
		path_aug_merge = self.aug_merge_path
		if len(trains) != len(labels) or len(trains) == 0 or len(trains) == 0:
			print("trains can't match labels")
			return 0
		for i in range(len(trains)):
			img_t = load_img(path_train+"/"+str(i)+"."+imgtype)
			img_l = load_img(path_label+"/"+str(i)+"."+imgtype)
			x_t = img_to_array(img_t)
			x_l = img_to_array(img_l)
			x_t[:,:,2] = x_l[:,:,0]
			img_tmp = array_to_img(x_t)
			img_tmp.save(path_merge+"/"+str(i)+"."+imgtype)
			img = x_t
			img = img.reshape((1,) + img.shape)
			savedir = path_aug_merge + "/" + str(i)
			if not os.path.lexists(savedir):
				os.mkdir(savedir)
			self.doAugmentate(img, savedir, str(i))


	def doAugmentate(self, img, save_to_dir, save_prefix, batch_size=1, save_format='tif', imgnum=30):

		"""
		augmentate one image
		"""
		datagen = self.datagen
		i = 0
		for batch in datagen.flow(img,
                          batch_size=batch_size,
                          save_to_dir=save_to_dir,
                          save_prefix=save_prefix,
                          save_format=save_format):
		    i += 1
		    if i > imgnum:
		        break

	def splitMerge(self):

		"""
		split merged image apart
		"""
		path_merge = self.aug_merge_path
		path_train = self.aug_train_path
		path_label = self.aug_label_path
		for i in range(self.slices): #slices = the number of training imgs read in
			path = path_merge + "/" + str(i)
			train_imgs = glob.glob(path+"/*."+self.img_type)
			savedir = path_train + "/" + str(i)
			if not os.path.lexists(savedir):
				os.mkdir(savedir)
			savedir = path_label + "/" + str(i)
			if not os.path.lexists(savedir):
				os.mkdir(savedir)
			for imgname in train_imgs:
				midname = imgname[imgname.rindex("/")+1:imgname.rindex("."+self.img_type)]
				img = cv2.imread(imgname)
				img_train = img[:,:,2]#cv2 read image rgb->bgr
				img_label = img[:,:,0]
				cv2.imwrite(path_train+"/"+str(i)+"/"+midname+"_train"+"."+self.img_type,img_train)
				cv2.imwrite(path_label+"/"+str(i)+"/"+midname+"_label"+"."+self.img_type,img_label)

	def splitTransform(self):

		"""
		split perspective transform images
		"""
		#path_merge = "transform"
		#path_train = "transform/data/"
		#path_label = "transform/label/"
		path_merge = "deform/deform_norm2"
		path_train = "deform/train/"
		path_label = "deform/label/"
		train_imgs = glob.glob(path_merge+"/*."+self.img_type)
		for imgname in train_imgs:
			midname = imgname[imgname.rindex("/")+1:imgname.rindex("."+self.img_type)]
			img = cv2.imread(imgname)
			img_train = img[:,:,2]#cv2 read image rgb->bgr
			img_label = img[:,:,0]
			cv2.imwrite(path_train+midname+"."+self.img_type,img_train)
			cv2.imwrite(path_label+midname+"."+self.img_type,img_label)
'''


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
	'''
	def create_train_data(self):
		"""
		Load training images and mask(label) images.
		Turn them into numpy arrays and save them in a list.
		Save them into .npy files.
		"""
		print('-'*30)
		print("Entering function: def create_train_data(self):")
		print('Loading training images and masks...')

		i = 0
		# Load the 30 .tif images.
		imgs = glob.glob(self.train_path+"/*."+self.img_type)
		print("# of imgs found: " + str(len(imgs)))
		# Create empty placeholder numpy array. (dim here: 30,512,512,1)
		imgdatas = np.ndarray((len(imgs),self.out_rows,self.out_cols,1), dtype=np.uint8)
		imglabels = np.ndarray((len(imgs),self.out_rows,self.out_cols,1), dtype=np.uint8)

		# loops thru imgs
		for imgname in imgs:
			midname = imgname[imgname.rindex("/")+1:]

			# load_img, img_to_array from keras
			img = load_img(self.train_path + "/" + midname,grayscale = True)
			label = load_img(self.label_path + "/" + midname,grayscale = True)
			img = img_to_array(img)
			label = img_to_array(label)
			
			# pad 0's because Conv2D cannot be allowed to get .5 values, as the upconv will not be same dim then
			img = np.pad(img, ((22,22),(0,0),(0,0)), 'constant')
			label = np.pad(label, ((22,22),(0,0),(0,0)), 'constant')

			# Zero mean and unit standard deviatio
			#sc = StandardScaler()
			#img = sc.fit_transform(img)
	        #imgs_test = sc.transform(imgs_test)

			#print("label shape: ")
			#print(label.shape)
			#label_inverted = 1 - label #didnt change anything. model predicts all black anyway

			# turned the img and label into numpy array of size (512,512,1)
			# populate list with img array after img array
			print(img.shape * 30)
			imgdatas[i] = img
			imglabels[i] = label#label_inverted
			i += 1
			# Can save created numpy arrays as imgs to see whether they are fine. Looks fine so far.			
			#img3 = array_to_img(img)
			#img3.save("./tests/%d.jpg" % (i))
			#img4 = array_to_img(label)
			#img4.save("./tests/l%d.jpg" % (i))

		print('Loaded {0}/{1} images.'.format(i, len(imgs)))

		np.save(self.npy_path + '/imgs_train.npy', imgdatas)
		np.save(self.npy_path + '/imgs_mask_train.npy', imglabels)
		print('Saving to .npy files done.')

		print("Finished function: def create_train_data(self):")
		print('-'*30)
		#return sc
	'''

	def create_train_data_nii(self):
		"""
		Load training images and mask(label) images.
		Turn them into numpy arrays, apply zero mean and unit standard deviation
		 and save them in a list.
		Save them into .npy files.
		"""
		print('-'*30)
		print('Loading training images and masks...')

		augment_type = 2 # 2=flip lr ud lrud. 1=rot3times

		i = 0
		# Load the images.
		parent_folder = './TumourData/Kurtosis_Gliomas_nii/'
		patient_folders = os.listdir(parent_folder)

		numOfImagesTotal = 0
		for patient_folder in patient_folders:
			image_nii = nibabel.load(parent_folder + patient_folder + '/' + 'flair_noskull.nii.gz')
			_, _, num_of_slices = image_nii.shape
			numOfImagesTotal += num_of_slices
		#print("num of slices total: " + str(numOfImagesTotal))
		
		if augment_type == 1:
			numOfImagesTotal = numOfImagesTotal * 4
		if augment_type == 2:
			numOfImagesTotal = numOfImagesTotal * 4

		# Create empty placeholder numpy array. (dim here: 30,512,512,1)
		imgdatas = np.ndarray((numOfImagesTotal,self.out_rows,self.out_cols, 1), dtype=np.float32) 
		imglabels = np.ndarray((numOfImagesTotal,self.out_rows,self.out_cols, 1), dtype=np.float32)#TODO: float32/uint8
		
		if i == 0:
			print("# of folders/patients should be 35 (37-2test): " + str(len(patient_folders)))
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
				print("SHIT'S ON FIRE, YO! Imgs and mask numbers not equal."*30)
			
			image_numpy = (image_numpy - mean(image_numpy)) / std(image_numpy)
			#print("image_numpy in data.py Max: " + str(np.max(np.array(image_numpy))))
			#print("image_numpy in data.py Min (negative): " + str(np.min(np.array(image_numpy))))
				


			for x in range(0, c1):
				# zero mean and unit standard deviation
				#image_numpy[:,:,x] = (image_numpy[:,:,x] - mean(image_numpy[:,:,x])) / std(image_numpy[:,:,x])
				
				if augment_type == 0:
					# We lose a dimension when we take out one slice ie [:,:,x]. 
					# Return to dim1xdim2x1 shape via img_to_array()
					img_train = img_to_array(image_numpy[:,:,x])
					mask = img_to_array(image_numpy_mask[:,:,x])
					
					imgdatas[i]  = img_train
					imglabels[i] = mask
					i += 1
				elif augment_type == 1:
					# We lose a dimension when we take out one slice. Return to dim1xdim2x1 shape.
					img_train = img_to_array(image_numpy[:,:,x])
					img_train_rota1 = img_to_array( np.rot90(image_numpy[:,:,x]) )
					img_train_rota2 = img_to_array( np.rot90(image_numpy[:,:,x], 2) )
					img_train_rota3 = img_to_array( np.rot90(image_numpy[:,:,x], 3) )

					mask = img_to_array(image_numpy_mask[:,:,x])
					mask_rota1 = img_to_array( np.rot90(image_numpy_mask[:,:,x]) )
					mask_rota2 = img_to_array( np.rot90(image_numpy_mask[:,:,x], 2) )
					mask_rota3 = img_to_array( np.rot90(image_numpy_mask[:,:,x], 3) )

					imgdatas[i]  = img_train
					imglabels[i] = mask
					i += 1
					imgdatas[i]  = img_train_rota1
					imglabels[i] = mask_rota1
					i += 1
					imgdatas[i]  = img_train_rota2
					imglabels[i] = mask_rota2
					i += 1
					imgdatas[i]  = img_train_rota3
					imglabels[i] = mask_rota3
					i += 1
				elif augment_type == 2:
					# We lose a dimension when we take out one slice. Return to dim1xdim2x1 shape.
					img_train 		  = img_to_array(image_numpy[:,:,x])
					img_train_fliplr  = img_to_array( np.fliplr(image_numpy[:,:,x]) ) # left right flip
					img_train_flipud  = img_to_array( np.flipud(image_numpy[:,:,x]) ) # up down flip
					img_train_fliplr_ud = img_to_array( np.flipud(np.fliplr(image_numpy[:,:,x])) ) # flip upDown AND leftRight

					mask 		   = img_to_array( image_numpy_mask[:,:,x] )
					mask_fliplr    = img_to_array( np.fliplr(image_numpy_mask[:,:,x]) ) # left right flip
					mask_flipud    = img_to_array( np.flipud(image_numpy_mask[:,:,x]) ) # up down flip
					mask_fliplr_ud = img_to_array( np.flipud(np.fliplr(image_numpy_mask[:,:,x])) ) # flip upDown AND leftRight

					imgdatas[i]  = img_train
					imglabels[i] = mask
					i += 1
					imgdatas[i]  = img_train_fliplr
					imglabels[i] = mask_fliplr
					i += 1
					imgdatas[i]  = img_train_flipud
					imglabels[i] = mask_flipud
					i += 1
					imgdatas[i]  = img_train_fliplr_ud
					imglabels[i] = mask_fliplr_ud
					i += 1



				_, _, channel = img_train.shape
				_, _, channel_mask = mask.shape
				if channel != 1:
					print("!!! Wrong Channel! !!! e444")
				if channel_mask != 1:
					print("!!! Wrong Channel! !!! e445")
			

		print('Loaded {0}/{1} images.'.format(i, numOfImagesTotal))

		np.save(self.npy_path + '/imgs_train.npy', imgdatas)
		np.save(self.npy_path + '/imgs_mask_train.npy', imglabels)

		print("imgdatas.shape: " + str(imgdatas.shape))
		print("imglabels.shape: " + str(imglabels.shape))
		print('Saved train-data and masks to .npy file.')
		print('-'*30)

	def create_test_data_nii(self):
		print('-'*30)
		print("Entering function: Them who say Niii: def create_test_data_nii(self):")
		print('Creating test images...')

		i = 0
		# Load the images.
		parent_folder = './TumourData/test_nii/'
		patient_folders = os.listdir(parent_folder)

		numOfImagesTotal = 0
		for patient_folder in patient_folders:
			image_nii = nibabel.load(parent_folder + patient_folder + '/' + 'flair_noskull.nii.gz')
			_, _, num_of_slices = image_nii.shape
			numOfImagesTotal += num_of_slices
		#print("num of slices total: " + str(numOfImagesTotal))
		
		# Create empty placeholder numpy array. (dim here: 30,512,512,1)
		imgdatas = np.ndarray((numOfImagesTotal,self.out_rows,self.out_cols, 1), dtype=np.float32) #MondayNightTODO was ,1 dim
		
		if i == 0:
			print("# of folders/patients should be 2 (37-35train): " + str(len(patient_folders)))
		# loops thru imgs
		for patient_folder in patient_folders:
			image_nii   = nibabel.load(parent_folder + patient_folder + '/' + 'flair_noskull.nii.gz')
			image_numpy = image_nii.get_data()
			#image_numpy = np.pad(image_numpy, ((22,22),(0,0),(0,0)), 'constant')
			a1, b1, c1 = image_numpy.shape
			print("=== === Img Shape before crop: " + str(image_numpy.shape))
			a1, b1, c1 = image_numpy.shape
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
	
	'''
	def create_test_data(self):
		print('-'*30)
		print("Entering function: def create_test_data(self):")
		print('Creating test images...')

		i = 0
		imgs = glob.glob(self.test_path+"/*."+self.img_type)
		print("# of images found: " + str(len(imgs)))
		# preallocate array.
		imgdatas = np.ndarray((len(imgs),self.out_rows,self.out_cols,1), dtype=np.uint8)

		for imgname in imgs:
			midname = imgname[imgname.rindex("/")+1:]
			img = load_img(self.test_path + "/" + midname,grayscale = True)
			img = img_to_array(img)
			img = np.pad(img, ((22,22),(0,0),(0,0)), 'constant')
			#img = sc.transform(img)
			imgdatas[i] = img
			i += 1
			#img2 = array_to_img(img)
			#img2.save("./tests/%d.jpg" % (i))

		print('loading done')
		np.save(self.npy_path + '/imgs_test.npy', imgdatas)
		print('Saving to imgs_test.npy files done.')

		print("Finished function: def create_test_data(self):")
		print('-'*30)
	'''

	def load_train_data(self):
		print('-'*30)
		print('load train images...')
		print('-'*30)
		imgs_train = np.load(self.npy_path+"/imgs_train.npy")
		imgs_mask_train = np.load(self.npy_path+"/imgs_mask_train.npy")

		print("negative min?")
		print("imgs_train in data.py Max: " + str(np.max(np.array(imgs_train))))
		print("imgs_train in data.py Min: " + str(np.min(np.array(imgs_train))))

		imgs_train = imgs_train.astype('float32')
		imgs_mask_train = imgs_mask_train.astype('float32')
		# imgs are not scaled to 255 and masks are already 0 or 1
		#imgs_train /= 255
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
		print('load test images...')
		print('-'*30)
		imgs_test = np.load(self.npy_path+"/imgs_test.npy")
		imgs_test = imgs_test.astype('float32')
		#imgs_test /= 255
		print("ndim: " + str(imgs_test.ndim))
		print("imgs_test in data.py Max: " + str(np.max(np.array(imgs_test))))
		print("imgs_test in data.py Min: " + str(np.min(np.array(imgs_test))))
		#mean = imgs_test.mean(axis = 0)
		#imgs_test -= mean
		return imgs_test
# End dataProcess class.


if __name__ == "__main__":

	#aug = myAugmentation()
	#aug.Augmentation()
	#aug.splitMerge()
	#aug.splitTransform()
	print("\n")
	print('='*30)
	dim1 = 256
	dim2 = 256
	mydata = dataProcess(dim1,dim2)
	print("Calling function .create_train_data()")
	mydata.create_train_data_nii()
	print("Calling function .create_test_data()")
	mydata.create_test_data_nii()
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

