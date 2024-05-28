# LW-CTrans
The code for the paper "LW-CTrans: A lightweight hybrid network of CNN and Transformer for 3D medical image segmentation" submitted to IEEE TMI. <br />

## Usage

### 0. Installation
* Install LW-CTrans as below
  
```
cd LW-CTrans
pip install -e .
```

### 1.1 Pre-processing
All compared methods use the same pre-processing steps as nnUNet. <br />

#### Dataset folder structure
Datasets must be located in the `nnUNet_raw` folder (which you either define when installing nnU-Net or export/set every 
time you intend to run nnU-Net commands!).
Each segmentation dataset is stored as a separate 'Dataset'. Datasets are associated with a dataset ID, a three digit 
integer, and a dataset name (which you can freely choose): For example, Dataset005_Prostate has 'Prostate' as dataset name and 
the dataset id is 5. Datasets are stored in the `nnUNet_raw` folder like this:

    nnUNet_raw/
    ├── Dataset001_BrainTumour
    ├── Dataset002_Heart
    ├── Dataset003_Liver
    ├── Dataset004_Hippocampus
    ├── Dataset005_Prostate
    ├── ...

Within each dataset folder, the following structure is expected:

    Dataset001_BrainTumour/
    ├── dataset.json
    ├── imagesTr
    ├── imagesTs  # optional
    └── labelsTr


When adding your custom dataset, take a look at the [dataset_conversion](../lwctrans/dataset_conversion) folder and 
pick an id that is not already taken. IDs 001-010 are for the Medical Segmentation Decathlon.

- **imagesTr** contains the images belonging to the training cases. nnU-Net will perform pipeline configuration, training with 
cross-validation, as well as finding postprocessing and the best ensemble using this data. 
- **imagesTs** (optional) contains the images that belong to the test cases. nnU-Net does not use them! This could just 
be a convenient location for you to store these images. Remnant of the Medical Segmentation Decathlon folder structure.
- **labelsTr** contains the images with the ground truth segmentation maps for the training cases. 
- **dataset.json** contains metadata of the dataset.

#### Experiment planning and preprocessing
Given a new dataset, nnU-Net will extract a dataset fingerprint (a set of dataset-specific properties such as
image sizes, voxel spacings, intensity information etc). This information is used to design three U-Net configurations. 
Each of these pipelines operates on its own preprocessed version of the dataset.

The easiest way to run fingerprint extraction, experiment planning and preprocessing is to use:

```bash
LWCTrans_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity
```

Where `DATASET_ID` is the dataset id (duh). We recommend `--verify_dataset_integrity` whenever it's the first time 
you run this command. This will check for some of the most common error sources!

You can also process several datasets at once by giving `-d 1 2 3 [...]`. If you already know what U-Net configuration 
you need you can also specify that with `-c 3d_fullres` (make sure to adapt -np in this case!). For more information 
about all the options available to you please run `LWCTrans_plan_and_preprocess -h`.

### 1.2 Training
Training models is done with the `LWCTrans_train` command. The general structure of the command is:
```bash
LWCTrans_train DATASET_NAME_OR_ID UNET_CONFIGURATION FOLD [additional options, see -h]
```

UNET_CONFIGURATION is a string that identifies the requested U-Net configuration (defaults: 2d, 3d_fullres, 3d_lowres, 
3d_cascade_lowres). DATASET_NAME_OR_ID specifies what dataset should be trained on and FOLD specifies which fold of 
the 5-fold-cross-validation is trained.

### 1.3 Validation
Validation is also done with the `LWCTrans_train` command. The general structure of the command is:
```bash
LWCTrans_train DATASET_NAME_OR_ID UNET_CONFIGURATION FOLD -val
```



## Acknowledgements
Part of codes are reused from the nnU-Net. Thanks to Fabian Isensee for the codes of nnU-Net.