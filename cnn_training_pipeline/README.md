# uhh_cnn_trainer

- collect.py was used to collect training and validation data for imitation learning
- ImitationCNN.ipynb is the Jupyter notebook file used to train the CNN for imitation learning, using all of the camera images in the provided image folders (the training image folders were too large to upload to GitHub)
- ImitationCNN_Filtered.ipynb is similar to above, but filters the images so that one can specify the number of left, right, straight, or stopped datapoints from each folder to use (did not perform as well)
- LicensePlateCNN.ipynb is the jupyter notebook file used to generate and augment license plates and use them to train the character and number CNNs 
