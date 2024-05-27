import numpy as np
import pandas as pd
import SimpleITK as sitk
from pathlib import Path
from skimage.measure import label, regionprops, regionprops_table
from scipy.stats import pearsonr


def get_diameter(arr: np.ndarray):
    label_img = label(arr)
    props = regionprops_table(label_img, properties=('axis_major_length', ))
    return props['axis_major_length'].max()


def get_volume(img_path):
    itk = sitk.ReadImage(img_path)
    arr = sitk.GetArrayFromImage(itk)
    arr[arr > 0] = 1
    unit_vol = np.prod(itk.GetSpacing())
    count = np.sum(arr)
    vol = count * unit_vol / 1000
    return vol


def main():
    pred_diameters = []
    gt_diameters = []
    for pred_path in pred_list:
        pat_name = pred_path.name
        gt_path = gt_basepath / pat_name
        pred = sitk.ReadImage(pred_path.as_posix())
        gt = sitk.ReadImage(gt_path.as_posix())
        pred_diameters.append(get_diameter(sitk.GetArrayFromImage(pred)))
        gt_diameters.append(get_diameter(sitk.GetArrayFromImage(gt)))
        # pred_diameters.append(get_volume(pred_path.as_posix()))
        # gt_diameters.append(get_volume(gt_path.as_posix()))
    res = pearsonr(pred_diameters, gt_diameters)
    # pd.DataFrame({'pred': pred_diameters, 'gt': gt_diameters}).to_excel('volume.xlsx', index=False)
    # print(gt_diameters)
    print(res)
    print(res.confidence_interval(confidence_level=0.95))


if __name__ == '__main__':
    data_path = [Path('/homeb/wyh/nnUNetv2/nnunetv2/results/Dataset420_LungCancer/nnUNetTrainer__nnUNetPlans__3d_fullres/TransHRNet_1/fold_0/validation')]
    gt_basepath = Path('/homeb/wyh/nnUNetv2/nnunetv2/data/nnUNet_preprocessed/Dataset420_LungCancer/gt_segmentations')

    for i in range(1):
        pred_list = data_path[i].glob('*.nii.gz')
        main()
