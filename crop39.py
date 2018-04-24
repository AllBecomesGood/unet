# Imports:
from PIL import Image
import os.path, sys

def crop(inputPath, outFolder, newWidth, newHeight):
	"""
	Crops images contained in specified folders to specified size.
	Equally crops all sides.
	Parameter: crop(inputPath, outFolder, newWidth, newHeight)
	inputPath eg './train/'
	outFolder eg './train256/'
	newWidth  eg 256
	newHeight eg 256
	"""
	path = inputPath
	dirs = os.listdir(path)
	for item in dirs:
		#print("item: " + item)
		fullpath = os.path.join(path, item)
		#print("fullpath: " + fullpath)

		if os.path.isfile(fullpath):
			im = Image.open(fullpath)
			f, e = os.path.splitext(fullpath)
			#print("e: " + e + " f: " + f)

			itemName, itemEnd = os.path.splitext(item)
			#print("itemName: " + itemName + " itemEnd: " + itemEnd)

			# Figure out how much to crop.
			width, height = im.size # width320, height276 for the MRI Data.
			if newHeight > height:
				print("Input image height smaller than target size. Can't crop, must enlarge!")
				print("Img in question: " + fullpath)
				break
			if newWidth > width:
				print("Input image width smaller than target size. Can't crop, must enlarge!")
				print("Img in question: " + fullpath)
				break

			
			# target size newWidth*newHeight
			excessWidth  = width  - newWidth
			excessHeight = height - newHeight

			#print(outFolder + itemName + '_256' + itemEnd)
			# Save image as .tif in specified location, adding 256 to name.
			#Parameters:	box – The crop rectangle(left, upper, right, lower)-tuple.
			imCrop = im.crop((excessWidth/2, 
							excessHeight/2, 
							width - excessWidth/2, 
							height - excessHeight/2))

			outPathAndName = outFolder + itemName + '_W' + str(newWidth) + 'xH' + str(newHeight) + '.tif' 
			imCrop.save(outPathAndName, "TIFF", quality=100)

crop('./tt1/', './tt2/', 128, 256)
#crop('./train/', './train256/', 256, 256)
#crop('./mask/', './mask256/', 256, 256)
#crop('./test/', './test256/', 256, 256)
#crop('./test_mask/', './test_mask256/', 256, 256)
