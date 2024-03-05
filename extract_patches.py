import random # import the random module
# from skimage import io # import the io module from skimage library
import numpy as np # import numpy library as np
from glob import glob # import the glob module from glob library
import SimpleITK as sitk # import the SimpleITK library as sitk
from keras.utils import np_utils # import np_utils from keras.utils library

class Pipeline(object): # create a class named Pipeline

    def __init__(self, train_list, Normalise=True): # define the constructor of the Pipeline class with train_list and Normalise as parameters
        self.scans_train = train_list # assign the train_list to scans_train variable of the instance
        self.train_imr = self.scans_read(Normalise) # call scans_read function and assign the result to train_imr variable of the instance


    def scans_read(self, Normalise):
        
        # Initialize an empty list called train_imr
        train_imr = []
        
        # Loop through the range of the length of scans_train and perform the following operations
        for i in range(len(self.scans_train)):
            
            # If i is divisible by 10 with no remainder, print the string 'iterations [{}]'
            if i % 10 == 0:
                print('iterations [{}]'.format(i))
            
            # Use glob to obtain file paths for flair, t2, segmentation, t1, and t1ce files for the current patient
            flairs = glob(self.scans_train[i] + '/*_flairs.nii.gz')
            t_2 = glob(self.scans_train[i] + '/*_t_2.nii.gz')
            g_t = glob(self.scans_train[i] + '/*_seg.nii.gz')
            t_1 = glob(self.scans_train[i] + '/*_t_1.nii.gz')
            t_1c = glob(self.scans_train[i] + '/*_t_1ce.nii.gz')
            
            # Create a list of t1 scans that are not t1ce
            t_1s = [scan for scan in t_1 if scan not in t_1c]
            
            # If the total number of scans for the current patient is less than 5, print a message indicating that there is a problem and continue to the next patient
            if (len(flairs)+len(t_2)+len(g_t)+len(t_1s)+len(t_1c)) < 5:
                print("here is the problem ,the problem lies in the patient :", self.scans_train[i])
                continue
            
            # Create a list of scans for the current patient in the order flair, t1, t1ce, t2, segmentation
            scans = [flairs[0], t_1s[0], t_1c[0], t_2[0], g_t[0]]
            
            # Use SimpleITK to read in the scans and convert them to arrays
            temporary = [sitk.GetArrayFromImage(sitk.ReadImage(scans[k])) for k in range(len(scans))]
            
            # Set the values for the start and end indices for z, y, and x axes to crop the arrays
            z_0 = 1
            y_0 = 29
            x_0 = 42
            z_1 = 147
            y1 = 221
            x_1 = 194
            
            # Convert the temporary array to a numpy array and crop it according to the previously defined indices
            temporary = np.array(temporary)
            temporary = temporary[:, z_0:z_1, y_0:y1, x_0:x_1]
            
            # If Normalise is True, normalize the slices using the self.normalise_slices method
            if Normalise == True:
                temporary = self.normalise_slices(temporary)
            
            # Append the temporary array to the train_imr list and delete it to save memory
            train_imr.append(temporary)
            del temporary
        
        # Return the train_imr list as a numpy array
        return np.array(train_imr)

    def randomly_patch_sample(self, number_patch, d, h, w):
        # Initialize empty lists for the patch and label arrays
        patch, label = [], []
        # Initialize a counter for the number of patches sampled
        counts = 0
        
        # Get the ground truth data for the images (assuming the fifth channel contains the ground truth data)
        g_t_imr = np.swapaxes(self.train_imr, 0, 1)[4]
        # Get the mask data for the images (assuming the first channel contains the mask data)
        mask = np.swapaxes(self.train_imr, 0, 1)[0]
        
        # Save the shape of the ground truth data array for reshaping later
        shp_temporary = g_t_imr.shape
        
        # Reshape the ground truth and mask data arrays to make it easier to sample random patches
        g_t_imr = g_t_imr.reshape(-1).astype(np.uint8)
        mask = mask.reshape(-1).astype(np.float32)
        
        # Find the indices in the mask array that correspond to valid (non-zero) pixels
        index = np.squeeze(np.argwhere((mask != -9.0) & (mask != 0.0)))
        # Delete the mask array to free up memory
        del mask
        
        # Randomly shuffle the index array
        np.random.shuffle(index)
        
        # Reshape the ground truth data array back to its original shape
        g_t_imr = g_t_imr.reshape(shp_temporary)
        
        # Initialize counters for iterating over the indices
        i = 0
        p_i_x = len(index)
        # Iterate over the indices until the desired number of patches have been sampled
        while (counts < number_patch) and (p_i_x > i):
            # Get the index at the current iteration
            ind_ = index[i]
            i += 1
            
            # Convert the index to the corresponding patient ID, slice index, and pixel coordinates
            ind_ = np.unravel_index(ind_, shp_temporary)
            patient_id = ind_[0]
            slice_index = ind_[1]
            p = ind_[2:]
            
            # Compute the coordinates of the patch based on the desired height and width
            PY = (p[0] - (h)/2, p[0] + (h)/2)
            PX = (p[1] - (w)/2, p[1] + (w)/2)
            PX = list(map(int, PX))
            PY = list(map(int, PY))
            
            # Get the patch data and corresponding ground truth label
            temporary = self.train_imr[patient_id][0:4,
                                                slice_index, PY[0]:PY[1], PX[0]:PX[1]]
            lbls = g_t_imr[patient_id, slice_index, PY[0]:PY[1], PX[0]:PX[1]]
            
            # If the patch is not the desired shape, skip to the next iteration
            if temporary.shape != (d, h, w):
                continue
            
            # Append the patch and label to the corresponding lists
            patch.append(temporary)
            label.append(lbls)
            # Increment the counter for the number of patches sampled
            counts += 1
        
        # Convert the patch and label lists to numpy arrays and return them
        patch = np.array(patch)
        label = np.array(label)
        return patch, label

    def normalise_slices(self, slice__not):
        """
        Normalize each slice in the input tensor using percentile-based intensity scaling,
        and return the normalized tensor.
        """
        slices_normed = np.zeros((5, 146, 192, 152)).astype(np.float32)

        # For each slice, apply normalization to each modality
        for sliceIX in range(4):
            # First, copy over the slice
            slices_normed[sliceIX] = slice__not[sliceIX]

            # Next, normalize each modality in the slice
            for modeIX in range(146):
                slices_normed[sliceIX][modeIX] = self._Normalise(
                    slice__not[sliceIX][modeIX])

        # Copy the last slice without normalizing
        slices_normed[-1] = slice__not[-1]

        return slices_normed


    def _Normalise(self, slice__):
        """
        Apply percentile-based intensity scaling to a single slice.
        """
        # Set the bottom and top percentiles for scaling
        b = np.percentile(slice__, 99)
        t = np.percentile(slice__, 1)

        # Clip the image intensities to the bottom and top percentiles
        slice__ = np.clip(slice__, t, b)

        # Calculate the standard deviation of the nonzero pixels in the slice
        non_zero_image = slice__[np.nonzero(slice__)]
        if np.std(slice__) == 0 or np.std(non_zero_image) == 0:
            # If the slice is completely empty or has no nonzero pixels,
            # return the input slice without normalization
            return slice__
        else:
            # Otherwise, normalize the slice by subtracting the mean of the nonzero pixels
            # and dividing by the standard deviation of the nonzero pixels.
            temporary = (slice__ - np.mean(non_zero_image)) / np.std(non_zero_image)

            # Set any values that become -inf after normalization to -9 (a sentinel value)
            temporary[temporary == temporary.min()] = -9
            return temporary


'''
def image_png_save(image, file_output="image.png"):

    # Convert the input image to a numpy array of float32 data type
    image = np.array(image).astype(np.float32)

    # Normalize the image so that the maximum value is 1
    if np.max(image) != 0:
        image /= np.max(image)

    # Scale the image so that the minimum value is -1, if necessary
    if np.min(image) <= -1:
        image /= abs(np.min(image))

    # Save the image as a PNG file
    io.imsave(file_output, image)

'''


def concatenation():

    # Load Y label and X patch data from both parts of the dataset
    Ylabel_2 = np.load("Ydata_set_second_part.npy").astype(np.uint8)
    Xpatch_2 = np.load("Xdata_set_second_part.npy").astype(np.float32)
    Ylabel_1 = np.load("Ydata_set_first_part.npy").astype(np.uint8)
    Xpatch_1 = np.load("Xdata_set_first_part.npy").astype(np.float32)

    # Concatenate the data from both parts along the 0th dimension
    Xpatch = np.concatenate((Xpatch_1, Xpatch_2), axis=0)
    Ylabel = np.concatenate((Ylabel_1, Ylabel_2), axis=0)

    # Delete the variables to free up memory
    del Ylabel_2, Xpatch_2, Ylabel_1, Xpatch_1

    # Combine the X and Y data into a list of tuples and shuffle it
    shuffles = list(zip(Xpatch, Ylabel))
    np.random.seed(138)
    np.random.shuffle(shuffles)

    # Separate the shuffled X and Y data back into separate arrays
    Xpatch = np.array([shuffles[i][0] for i in range(len(shuffles))])
    Ylabel = np.array([shuffles[i][1] for i in range(len(shuffles))])

    # Delete the shuffled list to free up memory
    del shuffles

    # Save the concatenated and shuffled X and Y data as numpy arrays
    np.save("Xtraining", Xpatch.astype(np.float32))
    np.save("Ytraining", Ylabel.astype(np.uint8))
    #np.save( "Xvalid",Xpatch_valid.astype(np.float32) )
    #np.save( "Yvalid",Ylabel_valid.astype(np.uint8))


if __name__ == '__main__':
    
    # Load paths of all HGG and LGG patients
    HGG_path = glob('Brats2017/Brats17TrainingData/HGG/**')
    LGG_path = glob('Brats2017/Brats17TrainingData/LGG/**')
    all_path = HGG_path+LGG_path

    # Shuffle the list of all paths
    np.random.seed(2022)
    np.random.shuffle(all_path)

    # Set patch size and number of patches
    h = 128
    w = 128
    d = 4
    number_patch = 146*(ending-starting)*3

    # Create pipeline object and sample random patches
    pipe = Pipeline(train_list=all_path[starting:ending], Normalise=True)
    patch, Ylabel = pipe.randomly_patch_sample(number_patch, d, h, w)

    # Transpose patch to correct order and convert labels to categorical format
    patch = np.transpose(patch, (0, 2, 3, 1)).astype(np.float32)
    Ylabel[Ylabel == 4] = 3
    shp = Ylabel.shape[0]
    Ylabel = Ylabel.reshape(-1)
    Ylabel = np_utils.to_categorical(Ylabel).astype(np.uint8)
    Ylabel = Ylabel.reshape(shp, h, w, 4)

    # Shuffle the patch and label arrays
    shuffles = list(zip(patch, Ylabel))
    np.random.seed(180)
    np.random.shuffle(shuffles)
    patch = np.array([shuffles[i][0] for i in range(len(shuffles))])
    Ylabel = np.array([shuffles[i][1] for i in range(len(shuffles))])
    del shuffles

    # Print the patch and label array sizes
    print("patch's Size : ", patch.shape)
    print("correponding target's Size   : ", Ylabel.shape)

    # Save patch and label arrays to disk as npy files
    #np.save( "Xdata_set_first_part",patch )
    #np.save( "Ydata_set_first_part",Ylabel)
