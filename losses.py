#import package
#import package numpy 
import numpy as np
# import keras
import keras.backend as K

# Define a function to calculate dice score for two input tensors
def dice(y_tru, y_predi):
    # Calculate the sums of the predicted and ground truth tensors
    sum_pred=K.sum(y_predi,axis=0)
    sum_rp=K.sum(y_tru,axis=0)
    # Calculate the sum of their element-wise multiplication
    sum_predr=K.sum(y_tru * y_predi,axis=0)
    # Calculate the numerator and denominator of the Dice score
    dice_numera =2*sum_predr
    dice_denomi =sum_rp+sum_pred
    # calculate dice score tensor and return it
    dice_score_tensor =(dice_numera+K.epsilon() )/(dice_denomi+K.epsilon())
    return dice_score_tensor

# Define a function to calculate dice score for the whole tumor
def dice_whole_tumor_metric(y_tru, y_predi):
    # Reshape the input tensors for easier manipulation
    y_tru_f = K.reshape(y_tru,shape=(-1,4))
    y_prediction_f = K.reshape(y_predi,shape=(-1,4))
    # Calculate the sums of the second to fourth channels of the tensors
    y_whole=K.sum(y_tru_f[:,1:],axis=1)
    p_whole=K.sum(y_prediction_f[:,1:],axis=1)
    # calculate dice score and return it
    dice_whole_tumor=dice(y_whole,p_whole)
    return dice_whole_tumor

# Define a function to calculate dice score for the enhancing tumor
def dice_enhancingancing_metric(y_tru, y_predi):
    # Reshape the input tensors for easier manipulation
    y_tru_f = K.reshape(y_tru,shape=(-1,4))
    y_prediction_f = K.reshape(y_predi,shape=(-1,4))
    # Calculate the sums of the fourth channel of the tensors
    y_enhancing=y_tru_f[:,-1]
    p_enhancing=y_prediction_f[:,-1]
    # calculate dice score and return it
    dice_enhancing=dice(y_enhancing,p_enhancing)
    return dice_enhancing

# Define a function to calculate dice score for the core region
def dice_core_region_metric(y_tru, y_predi):
    # Reshape the input tensors for easier manipulation
    y_tru_f = K.reshape(y_tru,shape=(-1,4))
    y_prediction_f = K.reshape(y_predi,shape=(-1,4))
    # Calculate the sums of the second and fourth channels of the tensors
    y_core_region=K.sum(y_tru_f[:,[1,3]],axis=1)
    p_core_region=K.sum(y_prediction_f[:,[1,3]],axis=1)
    # calculate dice score and return it
    dice_core_region=dice(y_core_region,p_core_region)
    return dice_core_region

# Define a function to calculate the weighted log loss
def weighted_log_loss(y_tru, y_predi):
    # Normalize the predicted tensor and clip it to avoid NaN values
    y_predi /= K.sum(y_predi, axis=-1, keepdims=True)
    y_predi = K.clip(y_predi, K.epsilon(), 1 - K.epsilon())
    # Define the weights for each class and calculate the loss
    weight=np.array([1,5,2,4])
    weight = K.variable(weight)
    loss = y_tru * K.log(y_predi) * weight
    loss = K.mean(-K.sum(loss, -1))
    return loss

def generate_dice_loss(y_tru, y_predi):
    # Reshape y_tru and y_predi to shape (-1, 4)
    y_tru_f = K.reshape(y_tru,shape=(-1,4))
    y_prediction_f = K.reshape(y_predi,shape=(-1,4))
    # Calculate sum of predictions, real positives and product of them
    sum_pred=K.sum(y_prediction_f,axis=-2)
    sum_rp=K.sum(y_tru_f,axis=-2)
    sum_predr=K.sum(y_tru_f * y_prediction_f,axis=-2)
    # Calculate the weight
    weight=K.pow(K.square(sum_rp)+K.epsilon(),-1)
    # Calculate the numerator and denominator of the generalised dice score tensor
    generalised_dice_numera =2*K.sum(weight*sum_predr)
    generalised_dice_denomi =K.sum(weight*(sum_rp+sum_pred))
    # Calculate the generalised dice score tensor
    generalised_dice_score_tensor =generalised_dice_numera /generalised_dice_denomi
    # Calculate the Generalized Dice Loss
    GDL=1-generalised_dice_score_tensor
    # Delete variables that are not needed anymore
    del sum_pred,sum_rp,sum_predr,weight
    # Return the sum of the Generalized Dice Loss and the weighted log loss
    return GDL+weighted_log_loss(y_tru,y_predi)

