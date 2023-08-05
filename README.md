# SHRED

This repo contains the code for various variants of the SHRED architecture.

A previous version of the repo, linked to our [MICCAI 2022 submission](http://rdcu.be/cVD6Y), can be found on the branch [`miccai_2022`](https://github.com/SCSE-Biomedical-Computing-Group/SHRED/tree/miccai_2022).

Data is available for download at the following links: [SchizConnect](http://schizconnect.org), [UCLA](https://openneuro.org/datasets/ds000030/versions/00016).
Some sites in SchizConnect seem to be down for some time.

## Environment Setup

1. Create and activate new conda environment

        conda create -n <env_name> python=3.8
        conda activate <env_name>

2. Run `setup.sh`

        chmod u+x ./setup.sh
        ./setup.sh
        
## Setup for a new dataset

1. Prepare dataset

    - Create a new folder under `./src` with the dataset name (see `./Schiz` for reference) and modify the setup and config files.
    - Edit `__init__.py` to specify how to retrieve site, age and gender. Labelling standards too, if applicable.
    - Add dataset to `DataloaderBase` class (`_get_indices()` too) and `Dataset` class in `./src/data.py`

2. Create `.yml` files in `config_template` to define model hyperparameters used and training settings. More details about the YAML files can be found in the [`miccai_2022`](https://github.com/SCSE-Biomedical-Computing-Group/SHRED/tree/miccai_2022) branch.

3. Train the model (and any other models - specify in the `.yml` file) using `single_stage_framework.py`.

        python single_stage_framework.py --config config_templates/individual/SHRED-III.yml
