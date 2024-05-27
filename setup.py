from setuptools import setup, find_namespace_packages

setup(name='lwctrans',
      packages=find_namespace_packages(include=["lwctrans", "lwctrans.*"]),
      version='2.1.1',
      description='nnU-Net. Framework for out-of-the box biomedical image segmentation.',
      url='https://github.com/MIC-DKFZ/nnUNet',
      author='Helmholtz Imaging Applied Computer Vision Lab, Division of Medical Image Computing, German Cancer Research Center',
      author_email='f.isensee@dkfz-heidelberg.de',
      license='Apache License Version 2.0, January 2004',
      python_requires=">=3.9",
      install_requires=[
          "torch>=2.0.0",
          "acvl-utils>=0.2",
          "dynamic-network-architectures>=0.2",
          "tqdm",
          "dicom2nifti",
          "scikit-image>=0.14",
          "scipy",
          "batchgenerators>=0.25",
          "numpy",
          "scikit-learn",
          "scikit-image>=0.19.3",
          "SimpleITK>=2.2.1",
          "pandas",
          "graphviz",
          'tifffile',
          'requests',
          "nibabel",
          "matplotlib",
          "seaborn",
          "imagecodecs",
          "yacs"
      ],
      entry_points={
          'console_scripts': [
              'LWCTrans_plan_and_preprocess = lwctrans.experiment_planning.plan_and_preprocess_entrypoints:plan_and_preprocess_entry',  # api available
              'LWCTrans_extract_fingerprint = lwctrans.experiment_planning.plan_and_preprocess_entrypoints:extract_fingerprint_entry',  # api available
              'LWCTrans_plan_experiment = lwctrans.experiment_planning.plan_and_preprocess_entrypoints:plan_experiment_entry',  # api available
              'LWCTrans_preprocess = lwctrans.experiment_planning.plan_and_preprocess_entrypoints:preprocess_entry',  # api available
              'LWCTrans_train = lwctrans.run.run_training:run_training_entry',  # api available
              'LWCTrans_predict_from_modelfolder = lwctrans.inference.predict_from_raw_data:predict_entry_point_modelfolder',  # api available
              'LWCTrans_predict = lwctrans.inference.predict_from_raw_data:predict_entry_point',  # api available
              'LWCTrans_convert_old_nnUNet_dataset = lwctrans.dataset_conversion.convert_raw_dataset_from_old_nnunet_format:convert_entry_point',  # api available
              'LWCTrans_find_best_configuration = lwctrans.evaluation.find_best_configuration:find_best_configuration_entry_point',  # api available
              'LWCTrans_determine_postprocessing = lwctrans.postprocessing.remove_connected_components:entry_point_determine_postprocessing_folder',  # api available
              'LWCTrans_apply_postprocessing = lwctrans.postprocessing.remove_connected_components:entry_point_apply_postprocessing',  # api available
              'LWCTrans_ensemble = lwctrans.ensembling.ensemble:entry_point_ensemble_folders',  # api available
              'LWCTrans_accumulate_crossval_results = lwctrans.evaluation.find_best_configuration:accumulate_crossval_results_entry_point',  # api available
              'LWCTrans_plot_overlay_pngs = lwctrans.utilities.overlay_plots:entry_point_generate_overlay',  # api available
              'LWCTrans_download_pretrained_model_by_url = lwctrans.model_sharing.entry_points:download_by_url',  # api available
              'LWCTrans_install_pretrained_model_from_zip = lwctrans.model_sharing.entry_points:install_from_zip_entry_point', # api available
              'LWCTrans_export_model_to_zip = lwctrans.model_sharing.entry_points:export_pretrained_model_entry', # api available
              'LWCTrans_move_plans_between_datasets = lwctrans.experiment_planning.plans_for_pretraining.move_plans_between_datasets:entry_point_move_plans_between_datasets',  # api available
              'LWCTrans_evaluate_folder = lwctrans.evaluation.evaluate_predictions:evaluate_folder_entry_point',  # api available
              'LWCTrans_evaluate_simple = lwctrans.evaluation.evaluate_predictions:evaluate_simple_entry_point',  # api available
              'LWCTrans_convert_MSD_dataset = lwctrans.dataset_conversion.convert_MSD_dataset:entry_point'  # api available
          ],
      },
      keywords=['deep learning', 'image segmentation', 'medical image analysis',
                'medical image segmentation', 'nnU-Net', 'nnunet']
      )
