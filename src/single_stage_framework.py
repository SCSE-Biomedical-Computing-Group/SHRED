import os
import json
import time
import yaml
import logging
import argparse
import pandas as pd
from itertools import product
from typing import Any, Dict

from data import Dataset
from config import EXPERIMENT_DIR, FrameworkConfigParser
from utils import mkdir, on_error, seed_torch
from factory import SingleStageFrameworkFactory
from trainer import SingleStageFrameworkTrainer, TrainerParams


@on_error(dict(), True)
def experiment(trainer: SingleStageFrameworkTrainer):
    trainer_results = trainer.run()
    return trainer_results.to_dict()


def process(config: Dict[str, Any]):
    seed_torch()
    logging.info("CONFIG:\n{}".format(json.dumps(config, indent=4)))

    script_name = os.path.splitext(os.path.basename(__file__))[0]
    experiment_name = "{}_{}_{}".format(
        script_name,
        int(time.strftime('%Y%m%d%H%M%S',time.localtime(time.time())) + str(time.time()).split('.')[-1][:3]),
        os.getpid()
    )

    output_dir = (
        config.get("output_directory", EXPERIMENT_DIR) or EXPERIMENT_DIR
    )
    output_dir = os.path.abspath(os.path.join(output_dir, experiment_name))

    config_path = os.path.join(
        output_dir, "{}.config.json".format(experiment_name),
    )
    results_path = os.path.join(output_dir, "{}.csv".format(experiment_name),)

    mkdir(output_dir)
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4, sort_keys=True)

    dataloader = SingleStageFrameworkFactory.load_dataloader(
        model_name=config["model_name"],
        dataloader_param={
            "dataset": Dataset(config["dataset"]),
            "harmonize": config["harmonize"],
        },
    )
    all_results = list()

    for seed, fold in product(config["seed"], config["fold"]):
        trainer = SingleStageFrameworkTrainer(
            dataloader=dataloader,
            trainer_params=TrainerParams(
                output_dir,
                config.get("model_name"),
                config.get("model_params", dict()),
                config.get("optim_params", dict()),
                config.get("hyperparameters", dict()),
                Dataset(config["dataset"]),
                seed,
                fold,
                config.get("ssl", False),
                config.get("harmonize", False),
                config.get("validation", False),
                config.get("labeled_sites", None),
                config.get("unlabeled_sites", None),
                config.get("num_unlabeled", None),
                config.get("device", -1),
                config.get("verbose", False),
                config.get("patience", float("inf")),
                config.get("max_epoch", 1000),
                config.get("save_model", False),
                config.get("dataloader_num_process", 10),
            ),
        )
        result = experiment(trainer)
        all_results.append(result)

        logging.info("RESULT:\n{}".format(json.dumps(result, indent=4)))

        df = pd.DataFrame(all_results).dropna(how="all")
        if not df.empty:
            df.to_csv(results_path, index=False)


def main(args):
    with open(os.path.abspath(args.config), "r") as f:
        configs: Dict[str, Any] = yaml.full_load(f)

    parser: FrameworkConfigParser = FrameworkConfigParser.parse(**configs)
    for config in parser.generate():
        process(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config_templates/single_stage_framework/config.yml",
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
