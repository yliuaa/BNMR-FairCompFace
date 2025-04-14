import torch
import pandas as pd
import numpy as np
import os
import torch.nn.functional as F
from tqdm import tqdm
import bnlearn as bn
import tabulate
from pgmpy.estimators import (
    BDeuScore,
    K2Score,
    BicScore,
    ExhaustiveSearch,
    MmhcEstimator,
)
from pgmpy.readwrite import XMLBIFWriter
from pgmpy.models import BayesianNetwork
from utils import *
from args import *


def generate_dataset():
    data_root = "../dataset"
    # Define datasets
    train_dataset = CelebA(
        data_root,
        split="train",
        target_type=["attr"],
        transform=transforms,
    )

    bnn_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=False)
    model = LightCNN_V4(args).to(args.device)
    model.load_state_dict(
        torch.load(
            f"../save/vanilla-lightCNN-{args.task}.pth",
            map_location=args.device,
        )
    )
    all_nodes = []
    for i, (data, attribute) in tqdm(
        enumerate(bnn_dataloader), total=len(bnn_dataloader), desc="Processing batches"
    ):
        inputs = data.to(args.device)
        output = model(inputs)
        probs = F.softmax(output, dim=1).squeeze().cpu().detach().numpy()
        attrs = attribute.squeeze().cpu().detach().numpy()
        preds = probs[:, 1].reshape(-1, 1)
        batch_nodes = np.hstack((attrs, preds))
        all_nodes.extend(batch_nodes)
    return all_nodes


if __name__ == "__main__":
    attribute_ids = make_attr_ids()
    data_file = f"init_bnn_data({args.task}_{attribute_ids}).csv"

    attribute_ids.extend([20, 39, 2, -9])
    if os.path.exists(data_file):
        data_bnn = pd.read_csv(data_file)

    else:
        attributes_df = pd.read_csv(
            "../dataset/celeba/list_attr_celeba.txt", delim_whitespace=True, header=1
        )
        columns = attributes_df.columns
        new_columns = columns.tolist() + [f"Model Prediction ({args.task})"]
        all_nodes = generate_dataset()
        data_bnn = pd.DataFrame(all_nodes, columns=new_columns)
        bin_edges = [i / 5 for i in range(6)]
        data_bnn[f"discritized prediction {args.task}"] = pd.cut(
            data_bnn[f"Model Prediction ({args.task})"], bins=bin_edges
        )
        data_bnn.to_csv(data_file)

    attributes_df = pd.read_csv(
        "../dataset/celeba/list_attr_celeba.txt", delim_whitespace=True, header=1
    )
    columns = attributes_df.columns
    columns_list = columns.to_list()
    target_ids = [columns_list[i] for i in attribute_ids]
    filtered_data = data_bnn[target_ids]
    filtered_data.replace({0: "False", 1: "True"}, inplace=True)

    """
        PGMPY learn
    """
    # bdeu = BDeuScore(filtered_data, equivalent_sample_size=5)
    # k2 = K2Score(filtered_data)
    # bic = BicScore(filtered_data)
    es = MmhcEstimator(filtered_data)

    best_model = es.estimate(significance_level=0.005)

    bnn = BayesianNetwork(best_model)
    bnn.fit(filtered_data)

    print("Best model: ")
    print(best_model.edges())
    print(len(best_model.edges()))

    writer = XMLBIFWriter(bnn)
    # writer.write_xmlbif(f"../save/(test)bnn_{args.task}_attr{attribute_ids}.xml")
    writer.write_xmlbif(f"../save/for_visualization.xml")
