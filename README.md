# Semi-supervised learning with data harmonisation for biomarker discovery from resting state fMRI

Code for SHRED-II is provided in this repository.

Models were developed in Python using Pytorch v1.8.0. The experiments were performed on an Nvidia P100 GPU.
- Information on sensitivity regarding parameter changes can be found in `./figures/`.
- Expected runtime: Depending on the size of the chosen site, a full run of 10 seeds x 5 folds for a single site can take between 1 to 3 minutes.
- Memory footprint: ～4GB of GPU memory

Data is available for download at the following links: [ABIDE](http://preprocessed-connectomes-project.org/abide/), [ADHD](http://preprocessed-connectomes-project.org/adhd200/).


## Environment Setup

1. Create and activate new Anaconda environment

        conda create -n <env_name> python=3.8
        conda activate <env_name>

2. Run ``setup.sh``

        chmod u+x ./setup.sh
        ./setup.sh

## Dataset Preparation

1. Process the raw fMRI data to obtain a functional connectivity matrix for each subject scan. The functional connectivity matrices should be saved as ``.npy`` files ([example](dataset/ABIDE/processed_corr_mat/)). There are no specific requirements on where the files should be saved at.

2. Prepare a CSV file ([example](dataset/ABIDE/meta.csv)) which contains the required columns below:

   - ``SUBJECT``: A unique identifier for different subjects
   - ``AGE``: The age of subjects when the fMRI scan is acquired
   - ``GENDER``: The gender of subjects
   - ``DISEASE_LABEL``: The label for disease classification (0 represents cognitive normal subjects, 1 represents diseased subjects)
   - ``SITE``: The site in which the subject scan is acquired
   - ``FC_MATRIX_PATH``: The paths to the ``.npy`` files that store the functional connectivity matrices of subjects. This path can either be absolute local path or relative path from the directory of the CSV file.

## Run the Code

1. Modify the ``config.yml`` file ([example](src/config.yml)) or create a new ``config.yml`` file as the input to the ``main.py`` script. The ``config.yml`` file contains the necessary arguments required to run the main script.

   - ``output_directory``: The directory in which the results should be stored at
   - ``model_name``: The name to be assigned to the model
   - ``model_params``: The parameters for initializing the EDC-VAE model, including
        - ``hidden_size``: The number of hidden nodes for encoder and decoder.
        - ``emb_size``: The dimension of the latent space representation output by the encoder.
        - ``clf_hidden_1``: The number of hidden nodes in the first hidden layer of classifier.
        - ``clf_hidden_2``:  The number of hidden nodes in the second hidden layer of classifier.
        - ``dropout``: The amount of dropout during training.
   - ``optim_params``: The parameters for initializing Adam optimizer
        - ``lr``: The learning rate
        - ``l2_reg``: The L2 regularization
   - ``hyperparameters``: Additional hyperparameters when training EDC-VAE model
        - ``ll_loss``: The weightage of VAE likelihood loss
        - ``kl_loss``: The weightage of KL divergence
   - ``dataset_path``: Path to the CSV prepared during dataset preparation stage. This path can be an absolute path, or a relative path from the directory of ``main.py``
   - ``dataset_name``: The name to be assigned to the dataset
   - ``seeds``: A list of seeds to iterate through
   - ``num_fold``: The number of folds for cross validation
   - ``ssl``: A boolean, indicating whether unlabeled data should be used to train the EDC-VAE model
   - ``harmonize``: A boolean, indicating whether ComBat harmonization should be performed prior to model training.
   - ``labeled_sites``: The site used as labeled data, when ``null``, all sites are used as labeled data.
   - ``device``: The GPU to be used, ``-1`` means to use CPU only
   - ``verbose``: A boolean, indicating Whether to display training progress messages
   - ``max_epoch``: The maximum number of epochs
   - ``patience``: The number of epochs without improvement for early stopping
   - ``save_model``: A boolean, indicating whether to save EDC-VAE's state_dict.

2. Run the main script

        cd src
        python main.py --config <PATH_TO_YML_FILE>

3. Load a saved model and perform inference

        python evaluate.py
