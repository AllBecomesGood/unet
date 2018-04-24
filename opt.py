import keras.backend as K
def dice_coef(y_true, y_pred, smooth, thresh):
    y_pred = y_pred > thresh
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)

    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(smooth, thresh):
	return dice

def dice(y_true, y_pred):
		return -dice_coef(y_true, y_pred, smooth, thresh)



# https://stackoverflow.com/questions/45961428/make-a-custom-loss-function-in-keras?noredirect=1&lq=1
def dicee(y_true, y_pred): 
	return -dice_coeff(y_true, y_pred, 1e-5, 0.5) 

def dice_coeff(y_true, y_pred, smooth, thresh): 
	y_pred = K.cast(y_pred > thresh ,dtype=tf.float32) 
	y_true = K.cast(y_true > thresh, dtype=tf.float32) 
	y_true_f = K.flatten(y_true) 
	y_pred_f = K.flatten(y_pred) 
	intersection = K.sum(y_true_f * y_pred_f) 
	return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth) 

#Final_Model.compile(optimizer=opt, loss=dice,metrics=['acc'])		




def dice_coef2(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef2(y_true, y_pred)

