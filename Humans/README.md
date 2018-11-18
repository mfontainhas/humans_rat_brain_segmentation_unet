# Deep Learning to segment different brain tissues in Humans

## Files Description:
The objective of each file is:
* Arq_ClaBrain.py - Architectures used for first classification of brain vs no brain
* Arq_SegBrain.py - Architectures used for second phase, segmentation of the brain
* Arq_WGM.py - Architectures used for third phase: Segmentation of the White and Gray matter
* helpingtools.py - Creates a bar tool to load images in Managefiles
* loss_functions.py - Some loss functions created to see some metrics.
* Managefiles_classification.py - Prepares files and turns them in to tensors with the respective labbles ready for DL: used for first classification of brain vs no brain
* Managefiles_mask.py - Prepares files and turns them in to tensors with the respective labbles ready for DL: used for segmentation of the brain using only-brain layers
* Managefiles_wgm.py - Prepares files and turns them in to tensors with the respective labbles ready for DL: used for Segmentation of the White and Gray matter using only-brain layers and all non-brain voxels set to zero
* NumpyImage.py - Pre-processing steps on MRI images
* test_Classification.py - File that tests Classification phase (it prints the lower and higher value of layer where the brain is inserted)
* test_flow_SegBrain.py - File that tests the Brain Segmentation, after the classification of brain vs no-brain layer pipeline. It uses the best models of Classification and Brain Segmentation
* test_flow_WGM.py - File that tests the White and Gray Matter segmentation, using Brain Segmentation and the classification of brain vs no-brain layer pipeline. It uses the best models of Classification, Brain Segmentation and WGM Segmentation
* train.py - It calls other functions. Creates the model of Arq_.py, Prepares the files with Managefiles_.py, and starts training. Compiles, defines epochs, loss functions, optmizers etc. And saves the final and best model of modelcheckpoint.
* writegraphs.py - Used to create the tensorboard graphs with validation and training lines in the same graph
* visualize.py - To save feature maps. Only used in the final after training, otherwise it would take a long time

Code by: Mariana Rodrigues
