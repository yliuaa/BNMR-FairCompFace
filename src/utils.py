import torch
from args import *
import pandas as pd
import numpy as np
from tqdm import tqdm
from model.lightCNN import LightCNN_V4
import torchvision.transforms as tfms
import torchvision
from torchvision.datasets import CelebA
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import Dataset, DataLoader, Subset
from pgmpy.readwrite import XMLBIFReader
from pgmpy.inference import VariableElimination


imagenet_mean = [
    0.485,
    0.456,
    0.406,
]
imagenet_std = [0.229, 0.224, 0.225]  # std of the ImageNet dataset for normalizing

transforms = tfms.Compose(
    [
        tfms.Resize((args.img_size, args.img_size)),
        tfms.ToTensor(),
        tfms.Normalize(imagenet_mean, imagenet_std),
    ]
)


def load_dataset(data_root, batch_size, v_batch_size, use_subset=True, debias=True):
    dataloaders = []

    # Define datasets
    train_dataset = CelebA(
        data_root,
        split="train",
        target_type=["attr", "landmarks"],
        transform=transforms,
    )
    test_dataset = CelebA(
        data_root,
        split="test",
        target_type=["attr", "landmarks"],
        transform=transforms,
    )
    valid_dataset = CelebA(
        data_root,
        split="valid",
        target_type=["attr", "landmarks"],
        transform=transforms,
    )

    if use_subset:
        # randomly sample subset
        train_sample = torch.randperm(len(train_dataset))[:50000]
        test_sample = torch.randperm(len(test_dataset))
        val_sample = torch.randperm(len(valid_dataset))
        train_dataset = Subset(train_dataset, train_sample)
        test_dataset = Subset(test_dataset, test_sample)
        valid_dataset = Subset(valid_dataset, val_sample)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    dataloaders = [train_dataloader, test_dataloader]
    if not debias:
        return dataloaders

    else:
        valid_dataloader = DataLoader(
            valid_dataset, batch_size=v_batch_size, shuffle=False
        )
        dataloaders.append(valid_dataloader)
        return dataloaders


def make_attr_ids(manually_specify=True):
    attributes_df = pd.read_csv(
        "~/FairCompFace/dataset/celeba/list_attr_celeba.txt",
        delim_whitespace=True,
        header=1,
    )
    all_attributes = attributes_df.columns.to_numpy()
    ids = []

    if manually_specify:
        ids = [1, 6, 7, 14, 24]
        return ids

    # Print validation results
    for i, attr in enumerate(all_attributes):
        if attr in demographics_attr:
            continue
        print(f"{i}: {attr}")
        ids.append(i)
    return ids


# NOTE: This function does not consider 0 true label
def demographic_parity_loss(pred, label, attribute):
    mask_a1 = (attribute == 1) & (label == 1)
    mask_a0 = (attribute == 0) & (label == 1)

    pred_a0 = (
        torch.sum(pred[mask_a0])
        if not torch.isnan(pred[mask_a0].mean())
        else torch.tensor(0.0)
    )
    pred_a1 = (
        torch.sum(pred[mask_a1])
        if not torch.isnan(pred[mask_a1].mean())
        else torch.tensor(0.0)
    )

    parity_diff = torch.abs(pred_a0 - pred_a1)
    return parity_diff


def convert_to_string(value):
    return "True" if value == 1 else "False" if value == 0 else None


def calibrated_demographic_parity_loss(
    bnet, pred, label, attributes, k, calibration=True
):
    a_k = attributes[:, k]
    ratio_0 = 1
    ratio_1 = 1
    mask_a1 = (a_k == 1) & (label == 1)
    mask_a0 = (a_k == 0) & (label == 1)
    positive_threshold = "_0_8_1_0_"
    if calibration:
        infer = VariableElimination(bnet)
        cpd = infer.query(
            variables=[attr_names[k]],
            evidence={"pred": positive_threshold},
        ).values
        prior = infer.query(variables=[attr_names[k]]).values
        with torch.no_grad():
            ratio_0 = torch.tensor(cpd[0] / prior[0])
            ratio_1 = torch.tensor(cpd[1] / prior[1])

    pred_a0 = (
        torch.mean(pred[mask_a0] * ratio_0)
        if not torch.isnan(pred[mask_a0].mean())
        else torch.tensor(0.0)
    )
    pred_a1 = (
        torch.mean(pred[mask_a1] * ratio_1)
        if not torch.isnan(pred[mask_a1].mean())
        else torch.tensor(0.0)
    )
    calibrated_parity_diff = torch.abs(pred_a0 - pred_a1)

    return calibrated_parity_diff


def mean_demographic_parity_loss(pred, label, feature_vector, cumulative=True):
    total_parity_diff = 0.0
    # calculate mean demogphaic parity difference across specified attributes
    for attr in range(feature_vector.shape[1]):
        mask_a1 = (feature_vector[:, attr] == 1) & (label == 1)
        mask_a0 = (feature_vector[:, attr] == 0) & (label == 1)

        pred_a0 = (
            torch.sum(pred[mask_a0])
            if not torch.isnan(pred[mask_a0].mean())
            else torch.tensor(0.0)
        )
        pred_a1 = (
            torch.sum(pred[mask_a1])
            if not torch.isnan(pred[mask_a1].mean())
            else torch.tensor(0.0)
        )

        total_parity_diff += torch.abs(pred_a0 - pred_a1)

    if cumulative:
        return pred_a0, pred_a1, sum(mask_a0), sum(mask_a1)

    else:
        return total_parity_diff / feature_vector.shape[1]


def softmax_normalization(X):
    exp_X = torch.exp(X / args.tau)
    return exp_X / torch.sum(exp_X)
    # return 1+ exp_X / torch.sum(exp_X)


def fair_eval(labels, all_predictions, fair_attributes):
    fairness = {}
    mask_a1 = (fair_attributes == 1) & (labels == 1)
    mask_a0 = (fair_attributes == 0) & (labels == 1)

    # just to know what's going on
    if sum(mask_a0) == 0 or sum(mask_a1) == 0:
        print("sample insufficient")

    p_pred_given_a1 = np.mean(all_predictions[mask_a1])
    p_pred_given_a0 = np.mean(all_predictions[mask_a0])

    # parity
    stat_parity = np.abs(p_pred_given_a1 - p_pred_given_a0)
    # Disparate Impact
    disparate_impact = (p_pred_given_a1 + 1e-6) / (p_pred_given_a0 + 1e-6)

    fairness["statistical_disparity"] = stat_parity
    fairness["disparate_impact"] = disparate_impact

    return fairness


def evaluate(
    model, dataloader, criterion, device, protected_attrs, target_id, dataset="valid"
):
    model.eval()
    test_loss = 0.0
    all_attrs = []
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for i, (inputs, (attributes, _)) in tqdm(
            enumerate(dataloader), total=len(dataloader), desc=f"{dataset} Evaluating"
        ):
            all_attrs.extend(attributes)
            labels = attributes[:, target_id].to(device)
            inputs = inputs.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.mean().item()

            _, predicted = outputs.max(1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    all_attrs = np.asarray(all_attrs)
    accuracy = accuracy_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)
    average_loss = test_loss / (i + 1)
    results = {
        f"{dataset}_accuracy": accuracy,
        f"{dataset}_f1_score": f1,
        f"{dataset}_loss": average_loss,
    }

    for attr in protected_attrs:
        fairness_metrics = fair_eval(
            all_labels, all_predictions, fair_attributes=all_attrs[:, attr]
        )
        results[f"{dataset}_disparate_impact_{attr}"] = fairness_metrics[
            "disparate_impact"
        ]
        results[f"{dataset}_statistical_disparity_{attr}"] = fairness_metrics[
            "statistical_disparity"
        ]

    return results


def build_model(model):
    if model == "lightCNN":
        return LightCNN_V4(args)


def adjust_for_numerical_stability(val):
    return val + 1e-7 if val == 0 else val


def cosine_similarity(A, B):
    dot_product = np.dot(A, B)
    norm_A = np.linalg.norm(A)
    norm_B = np.linalg.norm(B)
    return dot_product / (norm_A * norm_B)



def fetch_fair_loaders(data_root, attr_ids, transforms, group_size):
    valid_dataset = CelebA(
        data_root,
        split="valid",
        target_type=["attr", "landmarks"],
        transform=transforms,
    )

    positive_data = {key: [] for key in attr_ids}
    negative_data = {key: [] for key in attr_ids}
    for i, (data, (attribues, _)) in enumerate(valid_dataset):
        pos_mask = zip(attr_ids, attribues[attr_ids])
        for item in pos_mask:
            if item[1]:
                positive_data[item[0]].append((data, attribues))
            else:
                negative_data[item[0]].append((data, attribues))

    lengths = []
    for key, value in positive_data.items():
        num = len(value)
        print(f"{key}: {num}")
        lengths.append(num)
    print(" ============== ")
    for key, value in negative_data.items():
        num = len(value)
        print(f"{key}: {num}")
        lengths.append(num)
    assert group_size < min(lengths), "Subgroup size out of boundary."
    return positive_data, negative_data, min(lengths)


def sample_valid_data(positive_data, negative_data, group_size, attr_ids, length):
    sample_ids = torch.randperm(length)[:group_size].tolist()
    data = {
        key: [positive_data[key][i] for i in sample_ids]
        + [negative_data[key][i] for i in sample_ids]
        for key in attr_ids
    }
    return data


def add_dummy_rows(df, column, states):
    for state in states:
        if state not in df[column].unique():
            dummy_data = {col: [np.nan] for col in df.columns}
            dummy_data[column] = state
            df = pd.concat([df, pd.DataFrame(dummy_data)], ignore_index=True)
    return df



