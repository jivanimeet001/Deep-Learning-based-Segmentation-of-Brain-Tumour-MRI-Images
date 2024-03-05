import numpy as np
# import scipy
from scipy import ndimage

# Calculate binary dice coefficient for 3D images


def calculate_binary_dice_3d(s, g):
    numerator_ = np.sum(np.multiply(s, g))  # calculate the numerator
    denominator_ = s.sum() + g.sum()  # calculate the denominator
    if denominator_ == 0:  # if denominator is zero, return 1
        return 1
    else:
        # calculate and return the dice coefficient
        return 2.0 * numerator_ / denominator_

# Calculate sensitivity for 3D images


def calculate_sensitivity(segmentation, ground_truth):
    # calculate the numerator
    numerator_ = np.sum(np.multiply(ground_truth, segmentation))
    denominator_ = np.sum(ground_truth)  # calculate the denominator
    if denominator_ == 0:  # if denominator is zero, return 1
        return 1
    else:
        return numerator_ / denominator_  # calculate and return the sensitivity

# Calculate specificity for 3D images


def calculate_specificity(segmentation, ground_truth):
    # calculate the numerator
    numerator_ = np.sum(np.multiply(ground_truth == 0, segmentation == 0))
    denominator_ = np.sum(ground_truth == 0)  # calculate the denominator
    if denominator_ == 0:  # if denominator is zero, return 1
        return 1
    else:
        return numerator_ / denominator_  # calculate and return the specificity

# Create a border map for 3D binary images


def create_border_map(bin_img, neighbour):
    # convert the binary image to unsigned integer type
    bin_map = np.asarray(bin_img, dtype=np.uint8)
    neighbour = neighbour
    # shift the image along the west direction
    west_ = ndimage.shift(bin_map, [-1, 0, 0], order=0)
    # shift the image along the east direction
    east_ = ndimage.shift(bin_map, [1, 0, 0], order=0)
    # shift the image along the north direction
    north_ = ndimage.shift(bin_map, [0, 1, 0], order=0)
    # shift the image along the south direction
    south_ = ndimage.shift(bin_map, [0, -1, 0], order=0)
    # shift the image along the top direction
    top_ = ndimage.shift(bin_map, [0, 0, 1], order=0)
    # shift the image along the bottom direction
    bottom_ = ndimage.shift(bin_map, [0, 0, -1], order=0)
    cumulativ = west_ + east_ + north_ + south_ + \
        top_ + bottom_  # sum the shifted images
    # create the border map by comparing the sum with a threshold
    border_ = ((cumulativ < 6) * bin_map) == 1
    return border_  # return the border map

# This function calculates the Euclidean distance between the borders of the reference and segmentation images.


def calculate_border_distance(reference, segmentation):
    neighbour = 8
    # create border map for reference image
    border_reference = create_border_map(reference, neighbour)
    # create border map for segmentation image
    border_segmentation = create_border_map(segmentation, neighbour)
    # create the opposite of reference image
    oppose_reference = 1 - reference
    # create the opposite of segmentation image
    oppose_segmentation = 1 - segmentation
    # calculate the Euclidean distance transform for reference image
    distance_reference = ndimage.distance_transform_edt(oppose_reference)
    # calculate the Euclidean distance transform for segmentation image
    distance_segmentation = ndimage.distance_transform_edt(oppose_segmentation)
    # calculate the border distance between segmentation and reference images
    distance_border_segmentation = border_reference * distance_segmentation
    # calculate the border distance between reference and segmentation images
    distance_border_reference = border_segmentation * distance_reference

    # return the distances between the borders of segmentation and reference images
    return distance_border_reference, distance_border_segmentation


# This function calculates Hausdorff distance between the borders of the reference and segmentation images.
def calculate_hausdorff_distance(reference, segmentation):
    # calculate the border distances between segmentation and reference images
    reference_border_distance, segmentation_border_distance = calculate_border_distance(
        reference, segmentation)
    # calculate Hausdorff distance as the maximum of the border distances
    calculate_hausdorff_distance = np.max(
        [np.max(reference_border_distance), np.max(segmentation_border_distance)])

    # return Hausdorff distance between segmentation and reference images
    return calculate_hausdorff_distance


# This function calculates the Dice score for the whole tumor.
def calculate_dice_score_whole_tumor(pred_, origin_label):
    return calculate_binary_dice_3d(pred_ > 0, origin_label > 0)


# This function calculates the Dice score for the enhancing region.
def calculate_dice_score_enhancing_region(pred_, origin_label):
    return calculate_binary_dice_3d(pred_ == 4, origin_label == 4)


# This function computes the Dice score.
def compute_dice_core(pred_, origin_label):
    segmentation_ = np.copy(pred_)
    ground_truth_ = np.copy(origin_label)
    # set the voxels labeled as 2 (necrotic and non-enhancing tumor core) to 0
    segmentation_[segmentation_ == 2] = 0
    ground_truth_[ground_truth_ == 2] = 0
    # calculate the Dice score
    return calculate_binary_dice_3d(segmentation_ > 0, ground_truth_ > 0)


# This function computes the sensitivity for the whole tumor.
def compute_sensitivity_whole(segmentation, ground_truth):
    return calculate_sensitivity(segmentation > 0, ground_truth > 0)


# This function computes the sensitivity for the enhancing region.
def compute_sensitivity_en(segmentation, ground_truth):
    return calculate_sensitivity(segmentation == 4, ground_truth == 4)


# This function computes the sensitivity for the tumor core.
def compute_sensitivity_core(segmentation, ground_truth):
    segmentation_ = np.copy(segmentation)
    ground_truth_ = np.copy(ground_truth)
    # set the voxels labeled as 2 (necrotic and non-enhancing tumor core) to 0
    segmentation_[segmentation_ == 2] = 0
    ground_truth_[ground_truth_ == 2] = 0
    # calculate the sensitivity
    return calculate_sensitivity(segmentation_ > 0, ground_truth_ > 0)

# This function computes the specificity for the entire tumor region.
# It calls the calculate_specificity function with the segmentation and ground_truth arrays
# thresholded at 0 to convert them to binary arrays.


def compute_specificity_whole(segmentation, ground_truth):
    return calculate_specificity(segmentation > 0, ground_truth > 0)


# This function computes the specificity for the enhancing tumor region.
# It calls the calculate_specificity function with the segmentation and ground_truth arrays
# thresholded at 4 (enhancing tumor label) to convert them to binary arrays.
def compute_specificity_en(segmentation, ground_truth):
    return calculate_specificity(segmentation == 4, ground_truth == 4)


# This function computes the specificity for the core tumor region.
# It first creates copies of the segmentation and ground_truth arrays to avoid modifying the originals.
# It then sets all voxels with label 2 (non-enhancing tumor) to 0.
# It calls the calculate_specificity function with the modified segmentation and ground_truth arrays
# thresholded at 0 to convert them to binary arrays.
def compute_specificity_core(segmentation, ground_truth):
    segmentation_ = np.copy(segmentation)
    ground_truth_ = np.copy(ground_truth)
    segmentation_[segmentation_ == 2] = 0
    ground_truth_[ground_truth_ == 2] = 0
    return calculate_specificity(segmentation_ > 0, ground_truth_ > 0)


# This function computes Hausdorff distance for the entire tumor region.
# It calls the calculate_hausdorff_distance function with the segmentation and ground_truth arrays
# thresholded at 0 to convert them to binary arrays.
def compute_hausdorff_whole(segmentation, ground_truth):
    return calculate_hausdorff_distance(segmentation == 0, ground_truth == 0)


# This function computes Hausdorff distance for the enhancing tumor region.
# It calls the calculate_hausdorff_distance function with the segmentation and ground_truth arrays
# thresholded at values not equal to 4 (non-enhancing tumor or background) to convert them to binary arrays.
def compute_hausdorff_en(segmentation, ground_truth):
    return calculate_hausdorff_distance(segmentation != 4, ground_truth != 4)


# This function computes Hausdorff distance for the core tumor region.
# It first creates copies of the segmentation and ground_truth arrays to avoid modifying the originals.
# It then sets all voxels with label 2 (non-enhancing tumor) to 0.
# It calls the calculate_hausdorff_distance function with the modified segmentation and ground_truth arrays
# thresholded at 0 to convert them to binary arrays.
def compute_hausdorff_core(segmentation, ground_truth):
    segmentation_ = np.copy(segmentation)
    ground_truth_ = np.copy(ground_truth)
    segmentation_[segmentation_ == 2] = 0
    ground_truth_[ground_truth_ == 2] = 0
    return calculate_hausdorff_distance(segmentation_ == 0, ground_truth_ == 0)
