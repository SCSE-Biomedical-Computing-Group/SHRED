import torch
from typing import Any, Dict
from torch_geometric.data import Data

from data import Dataset
from utils.metrics import ClassificationMetrics as CM
from models.EDC_VAE import EDC_VAE


def load_model(model_params: Dict[str, Any], model_path: str):
    model = EDC_VAE(**model_params)
    model.load_state_dict(
        torch.load(model_path, map_location=torch.device("cpu"))
    )
    return model


def load_data(data_csv_path: str, harmonize: bool = False):
    dataset = Dataset(data_csv_path, "", harmonize)
    return dataset.load_all_data()["data"]


def evaluate_model(model: EDC_VAE, data: Data):
    x, y = data.x, data.y
    prediction = model.forward(x)
    print("accuracy: {:.5f}".format(CM.accuracy(y, prediction["y"]).item()))
    print("f1: {:.5f}".format(CM.f1_score(y, prediction["y"]).item()))
    print("recall: {:.5f}".format(CM.tpr(y, prediction["y"]).item()))
    print("precision: {:.5f}".format(CM.ppv(y, prediction["y"]).item()))


if __name__ == "__main__":
    model = load_model(
        dict(
            input_size=34716,
            hidden_size=32,
            emb_size=16,
            clf_hidden_1=0,
            clf_hidden_2=0,
        ),
        model_path="../saved_model/ABIDE_VAE-FFN_0_0_1645419832.pt"
    )
    data = load_data("../dataset/ABIDE/meta.csv")
    evaluate_model(model, data)