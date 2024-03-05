# Importing necessary libraries and modules
import os
from evaluation_metrics import *
import numpy as np
from model import Unet_model
from glob import glob
import SimpleITK as sitk
import random


class Prediction(object):

    def __init__(self, batch_size_test, load_model_path):

        self.batch_size_test = batch_size_test
        unet = Unet_model(img_shape=(240, 240, 4),
                          load_model_weights=load_model_path)
        self.model = unet.model
        print('U-net CNN compiled!\n')

    def predict_volume(self, filepath_image, show):

        flair = glob(filepath_image + '/*_flair.nii.gz')
        t2 = glob(filepath_image + '/*_t2.nii.gz')
        gt = glob(filepath_image + '/*_seg.nii.gz')
        t1s = glob(filepath_image + '/*_t1.nii.gz')
        t1c = glob(filepath_image + '/*_t1ce.nii.gz')
        t1 = [scan for scan in t1s if scan not in t1c]
        if (len(flair)+len(t2)+len(gt)+len(t1)+len(t1c)) < 5:
            print("there is a problem here!!! the problem lies in this patient :")
        scans_test = [flair[0], t1[0], t1c[0], t2[0], gt[0]]
        test_im = [sitk.GetArrayFromImage(sitk.ReadImage(
            scans_test[i])) for i in range(len(scans_test))]

        test_im = np.array(test_im).astype(np.float32)
        test_image = test_im[0:4]
        gt = test_im[-1]
        gt[gt == 4] = 3

        test_image = self.norm_slices(test_image)

        test_image = test_image.swapaxes(0, 1)
        test_image = np.transpose(test_image, (0, 2, 3, 1))

        if show:
            verbose = 1
        else:
            verbose = 0
        prediction = self.model.predict(
            test_image, batch_size=self.batch_size_test, verbose=verbose)
        prediction = np.argmax(prediction, axis=-1)
        prediction = prediction.astype(np.uint8)

        prediction[prediction == 3] = 4
        gt[gt == 3] = 4

        return np.array(prediction), np.array(gt)

    def evaluate_segmented_volume(self, filepath_image, save, show, save_path):

        predicted_images, gt = self.predict_volume(filepath_image, show)

        if save:
            tmp = sitk.GetImageFromArray(predicted_images)
            sitk.WriteImage(tmp, 'predictions/{}.nii.gz'.format(save_path))

        Dice_complete = calculate_dice_score_whole_tumor(predicted_images, gt)
        Dice_enhancing = calculate_dice_score_enhancing_region(
            predicted_images, gt)
        Dice_core = compute_dice_core(predicted_images, gt)

        Sensitivity_whole = compute_sensitivity_whole(predicted_images, gt)
        Sensitivity_en = compute_sensitivity_en(predicted_images, gt)
        Sensitivity_core = compute_sensitivity_core(predicted_images, gt)

        Specificity_whole = compute_specificity_whole(predicted_images, gt)
        Specificity_en = compute_specificity_en(predicted_images, gt)
        Specificity_core = compute_specificity_core(predicted_images, gt)

        Hausdorff_whole = compute_hausdorff_whole(predicted_images, gt)
        Hausdorff_en = compute_hausdorff_en(predicted_images, gt)
        Hausdorff_core = compute_hausdorff_core(predicted_images, gt)

        if show:
            print("************************************************************")
            print("Dice complete tumor score : {:0.4f}".format(Dice_complete))
            print(
                "Dice core tumor score (tt sauf vert): {:0.4f}".format(Dice_core))
            print("Dice enhancing tumor score (jaune):{:0.4f} ".format(
                Dice_enhancing))
            print("**********************************************")
            print("Sensitivity complete tumor score : {:0.4f}".format(
                Sensitivity_whole))
            print("Sensitivity core tumor score (tt sauf vert): {:0.4f}".format(
                Sensitivity_core))
            print("Sensitivity enhancing tumor score (jaune):{:0.4f} ".format(
                Sensitivity_en))
            print("***********************************************")
            print("Specificity complete tumor score : {:0.4f}".format(
                Specificity_whole))
            print("Specificity core tumor score (tt sauf vert): {:0.4f}".format(
                Specificity_core))
            print("Specificity enhancing tumor score (jaune):{:0.4f} ".format(
                Specificity_en))
            print("***********************************************")
            print("Hausdorff complete tumor score : {:0.4f}".format(
                Hausdorff_whole))
            print("Hausdorff core tumor score (tt sauf vert): {:0.4f}".format(
                Hausdorff_core))
            print("Hausdorff enhancing tumor score (jaune):{:0.4f} ".format(
                Hausdorff_en))
            print("***************************************************************\n\n")

        return np.array((Dice_complete, Dice_core, Dice_enhancing, Sensitivity_whole, Sensitivity_core, Sensitivity_en, Specificity_whole, Specificity_core, Specificity_en, Hausdorff_whole, Hausdorff_core, Hausdorff_en))  # ))

    def predict_multiple_volumes(self, filepath_volumes, save, show):

        results, Ids = [], []
        for patient in filepath_volumes:
            tmp1 = patient.split('/')
            print("Volume ID: ", tmp1[-2]+'/'+tmp1[-1])
            tmp = self.evaluate_segmented_volume(
                patient, save=save, show=show, save_path=os.path.basename(patient))

            results.append(tmp)

            Ids.append(str(tmp1[-2]+'/'+tmp1[-1]))

        res = np.array(results)
        print("mean : ", np.mean(res, axis=0))
        print("std : ", np.std(res, axis=0))
        print("median : ", np.median(res, axis=0))
        print("25 quantile : ", np.percentile(res, 25, axis=0))
        print("75 quantile : ", np.percentile(res, 75, axis=0))
        print("max : ", np.max(res, axis=0))
        print("min : ", np.min(res, axis=0))

        np.savetxt('Results.out', res)
        np.savetxt('Volumes_ID.out', Ids, fmt='%s')

    def norm_slices(self, slice_not):

        normed_slices = np.zeros((4, 155, 240, 240))
        for slice_ix in range(4):
            normed_slices[slice_ix] = slice_not[slice_ix]
            for mode_ix in range(155):
                normed_slices[slice_ix][mode_ix] = self._normalize(
                    slice_not[slice_ix][mode_ix])

        return normed_slices

    def _normalize(self, slice):

        b = np.percentile(slice, 99)
        t = np.percentile(slice, 1)
        slice = np.clip(slice, t, b)
        image_nonzero = slice[np.nonzero(slice)]

        if np.std(slice) == 0 or np.std(image_nonzero) == 0:
            return slice
        else:
            tmp = (slice - np.mean(image_nonzero)) / np.std(image_nonzero)
            tmp[tmp == tmp.min()] = -9
            return tmp


if __name__ == "__main__":

    model_to_load = "models_saved/ResUnet.04_0.646.hdf5"

    path_HGG = glob('Brats2017/Brats17TrainingData/HGG/**')
    path_LGG = glob('Brats2017/Brats17TrainingData/LGG/**')

    test_path = path_HGG+path_LGG
    np.random.seed(2022)
    np.random.shuffle(test_path)

    brain_seg_pred = Prediction(
        batch_size_test=2, load_model_path=model_to_load)

    brain_seg_pred.predict_multiple_volumes(
        test_path[200:290], save=False, show=True)
