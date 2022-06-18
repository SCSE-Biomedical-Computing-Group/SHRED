import os
import json
import time
import yaml
import logging
import argparse
import pandas as pd
from typing import Any, Dict

from utils.misc import mkdir, seed_torch
from trainer import EDC_VAE_Trainer, TrainerParams


def process(config: Dict[str, Any]):
    seed_torch()
    logging.info("CONFIG:\n{}".format(json.dumps(config, indent=4)))

    script_name = os.path.splitext(os.path.basename(__file__))[0]
    experiment_name = "{}_{}_{}".format(
        script_name, int(time.time()), os.getpid()
    )

    output_dir = config.get("output_directory")
    output_dir = os.path.abspath(os.path.join(output_dir, experiment_name))

    config_path = os.path.join(
        output_dir, "{}.config.json".format(experiment_name),
    )
    results_csv_path = os.path.join(
        output_dir, "{}.csv".format(experiment_name),
    )

    mkdir(output_dir)
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4, sort_keys=True)

    trainer = EDC_VAE_Trainer(
        trainer_params=TrainerParams(
            output_directory=output_dir,
            model_name=config.get("model_name"),
            model_params=config.get("model_params", dict()),
            optim_params=config.get("optim_params", dict()),
            hyperparameters=config.get("hyperparameters", dict()),
            dataset_path=config.get("dataset_path"),
            dataset_name=config.get("dataset_name"),
            seeds=config.get("seeds", list(range(10))),
            num_fold=config.get("num_fold", 5),
            ssl=config.get("ssl", False),
            harmonize=config.get("harmonize", False),
            labeled_sites=config.get("labeled_sites", None),
            device=config.get("device", -1),
            verbose=config.get("verbose", False),
            patience=config.get("patience", float("inf")),
            max_epoch=config.get("max_epoch", 1000),
            save_model=config.get("save_model", False),
        ),
    )
    trainer.run(results_csv_path)


def main(args):
    with open(os.path.abspath(args.config), "r") as f:
        config: Dict[str, Any] = yaml.full_load(f)
    process(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="./config.yml",
        help="the path to the config file",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG,
        format="[%(asctime)s] - %(filename)s: %(levelname)s: "
        "%(funcName)s(): %(lineno)d:\t%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main(args)
