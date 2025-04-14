import torch
import copy
from utils import *
from args import args
import warnings
import itertools
import datetime
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import pandas as pd
from tqdm.auto import tqdm
import torch.optim as optim
import torch.nn.functional as F
from metaSGD import *
from model.lightCNN import LightCNN_V4
from sklearn.metrics import f1_score, accuracy_score
from pgmpy.readwrite import XMLBIFReader
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.factors.discrete import TabularCPD
from pgmpy.sampling import BayesianModelSampling

# Constants
EXPECTED_STATES = ["_0_0_0_2_", "_0_2_0_4_", "_0_4_0_6_", "_0_6_0_8_", "_0_8_1_0_"]
PRIOR_SAMPLES = 80
attrs = ["True", "False"]
attr_ids = make_attr_ids()
print(len(attr_ids))
# Generate all possible combinations of these values
all_combinations = list(
    itertools.product(*([attrs] * len(attr_ids) + [EXPECTED_STATES]))
)
dummy_df = pd.DataFrame(
    all_combinations, columns=[attr_names[i] for i in attr_ids] + ["pred"]
)


# ---------------------------------------------------------------------------- #
#                    Bayesian Network Init and Update                          #
# ---------------------------------------------------------------------------- #

# Update belief network
def update_belief(bnn, batch_obs_df, init=False):
    # try:
    if init:
        combined_observation = pd.concat([dummy_df, batch_obs_df], ignore_index=True)
        estimator = MaximumLikelihoodEstimator(bnn, combined_observation)
        cpd_pred = estimator.estimate_cpd("pred")
        bnn.add_cpds(cpd_pred)
    else:
        prior_belief = BayesianModelSampling(bnn).likelihood_weighted_sample(
            size=PRIOR_SAMPLES
        )
        combined_observation = pd.concat(
            [dummy_df, prior_belief, batch_obs_df], ignore_index=True
        )
        estimator = MaximumLikelihoodEstimator(bnn, combined_observation)
        cpd_pred = estimator.estimate_cpd("pred")
        bnn.add_cpds(cpd_pred)
    # except ValueError:
    #     print("Bad data instances, skipped.")


def load_observations(bnn, batch_obs_df, outputs, attributes):
    bin_edges = [i / 5 for i in range(6)]
    probs = F.softmax(outputs, dim=1).squeeze().cpu().detach().numpy()
    preds = probs[:, 1].reshape(-1, 1)
    attr_arr = attributes.numpy()
    batch_observation = np.hstack((attr_arr, preds))
    new_batch = pd.DataFrame(batch_observation, columns=bnn.nodes)
    new_batch[f"pred"] = pd.cut(
        new_batch["pred"],
        bins=bin_edges,
        labels=EXPECTED_STATES,
    )
    new_batch.replace({float(0): "False", float(1): "True"}, inplace=True)
    if len(batch_obs_df) == 0:
        return new_batch
    else:
        new_observation = pd.concat([batch_obs_df, new_batch], ignore_index=True)
        return new_observation


def randomly_init_bnet(belief_net, dataloader, attr_ids):
    print("Initialize prediction node..")
    bin_edges = [i / 5 for i in range(6)]
    belief_net.add_node("pred")
    new_edges = [(node, "pred") for node in belief_net.nodes if node != "pred"]
    belief_net.add_edges_from(new_edges)
    _, (attributes, _) = next(iter(dataloader))
    pred = torch.linspace(start=0, end=1, steps=16).unsqueeze(1)
    attr_arr = attributes[:, attr_ids].numpy()
    batch_observation = np.hstack((attr_arr, pred))
    init_batch = pd.DataFrame(batch_observation, columns=belief_net.nodes)
    init_batch[f"pred"] = pd.cut(
        init_batch["pred"],
        bins=bin_edges,
        labels=EXPECTED_STATES,
    )
    init_batch.replace({float(0): "False", float(1): "True"}, inplace=True)
    update_belief(belief_net, init_batch, init=True)
    print("Finished initialize CPD for prediction!")


# ---------------------------------------------------------------------------- #
#                                 Loss Variants                                #
# ---------------------------------------------------------------------------- #
# def l_fair(model, data, device):
#     total_l_fair = torch.zeros(1, device=device)
#     for k in data.keys():
#         inputs = torch.stack([x[0] for x in data[k]])
#         attributes = [x[1] for x in data[k]]
#         attributes = torch.stack(attributes).to(device)
#         inputs = inputs.to(device)
#         labels = attributes[:, target_id]
#         output = model(inputs)
#         l_fair = demographic_parity_loss(output, labels, attributes[:, k])
#         wandb.log({f"attribute-{k}-parity": l_fair})
#         total_l_fair += l_fair
#     return total_l_fair


# Bayesian Net calibrated loss
def l_fair_calibrated(model, belief_net, data, device, calibration=True):
    total_l_fair = torch.zeros(1, device=device, requires_grad=True)
    for k in data.keys():
        inputs = torch.stack([x[0] for x in data[k]])
        attributes = [x[1] for x in data[k]]
        attributes = torch.stack(attributes).to(device)
        inputs = inputs.to(device)
        labels = attributes[:, target_id]
        outputs = model(inputs)
        probs = F.softmax(outputs, dim=1).squeeze()[:, 1]
        l_fair = calibrated_demographic_parity_loss(
            belief_net, probs, labels, attributes, k, calibration
        )
        total_l_fair = total_l_fair + l_fair
    total_l_fair /= args.subgroup_size
    return total_l_fair


# ---------------------------------------------------------------------------- #
#                                   Training                                   #
# ---------------------------------------------------------------------------- #
def train(args):
    data_root = args.data_root
    print(f"Load data from {data_root}")
    attr_ids = make_attr_ids()

    # -------------------- Preparing parameters and dataloader ------------------- #
    belief_net = None
    batch_size = args.batch_size
    v_batch_size = args.v_batch_size
    epochs = args.epochs
    lr = args.lr
    device = args.device
    criterion = torch.nn.CrossEntropyLoss(reduction="none")
    train_dataloader, test_dataloader, valid_dataloader = load_dataset(
        data_root, batch_size, v_batch_size, use_subset=True, debias=args.debias
    )
    pos_valid_data, neg_valid_data, length = fetch_fair_loaders(
        data_root, attr_ids, transforms, args.subgroup_size
    )
    print("Loadded pos/neg data")

    # ------------------------------- Load Models ------------------------------- #
    model = LightCNN_V4(args)
    # add (test) to enable belief update
    reader = XMLBIFReader(path=f"../save/(test)bnn[1, 6, 7, 14, 24].xml")

    if len(attr_ids):
        belief_net = reader.get_model()
    print(f"Loaded model at ../save/(test)bnn{attr_ids}.xml")
    print(args.online)
    if (
        args.method in ["Bayesian", "Bayesian_Online", "Bayesian_Finetune"]
        and args.online
    ):
        randomly_init_bnet(belief_net, train_dataloader, attr_ids)

    if args.load_model != "none":
        model.load_state_dict(
            torch.load(
                args.load_model,
                map_location=device,
            )
        )
    model.to(device)
    best_model = None
    best_val_acc = 0
    weights_t = torch.ones(args.batch_size, requires_grad=True, device=device)
    ones = torch.ones_like(weights_t)
    calibration = True

    # ----------------------------- Prepare optimizer ---------------------------- #
    optimizer = optim.Adam(model.parameters(), lr=lr)
    w_optimizer = optim.SGD([weights_t], lr=args.v_lr)

    # ------------------------------- Training Loop ------------------------------ #
    print(f"Start training with device {device}")
    validation_res = []
    for epoch in range(args.epoch, epochs):
        model.train()
        total_loss = 0.0
        fair_loss = 0.0
        all_predictions = []
        all_probs = []
        all_labels = []

        observations = pd.DataFrame([], columns=belief_net.nodes)
        for i, (inputs, (attributes, _)) in tqdm(
            enumerate(train_dataloader),
            total=len(train_dataloader),
            desc=f"Training Epoch: {epoch+1}/{epochs}",
            bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
        ):
            labels = attributes[:, target_id].to(device)
            inputs = inputs.to(device)
            if inputs.size()[0] != batch_size:
                print("Finished epoch")
                break

            # instantiate & update inner_model - code adapted from
            # https://github.com/ShiYunyi/Meta-Weight-Net_Code-Optimization/tree/main
            inner_model = build_model(args.model).to(device)
            inner_model.load_state_dict(model.state_dict())
            inner_preds = inner_model(inputs)
            if args.method == "Bayesian" and args.online:
                observations = load_observations(
                    belief_net, observations, inner_preds, attributes[:, attr_ids]
                )
                if i % 10 == 0:
                    observations = pd.DataFrame([], columns=belief_net.nodes)
                    belief_net.check_model()
                    update_belief(belief_net, observations)
            l_inner = (weights_t * criterion(inner_preds, labels)).mean()
            inner_grads = torch.autograd.grad(
                l_inner, (inner_model.parameters()), create_graph=True
            )
            pseudo_optimizer = MetaSGD(
                inner_model, inner_model.parameters(), lr=args.lr
            )
            pseudo_optimizer.load_state_dict(optimizer.state_dict())
            pseudo_optimizer.meta_step(inner_grads)
            del inner_grads

            # -------------------------------- Method Args ------------------------------- #

            if args.method in ["Bayesian", "BayesianInv", "Bayesian_Finetune", "FORML"]:
                if args.method == "FORML":
                    calibration = False
                else:
                    calibration = True
                sample_data = sample_valid_data(
                    pos_valid_data, neg_valid_data, args.subgroup_size, attr_ids, length
                )
                fair_loss = l_fair_calibrated(
                    inner_model, belief_net, sample_data, device, calibration
                )
                w_optimizer.zero_grad()
                fair_loss.backward()
                weights_t.data.copy_(ones)
                w_optimizer.step()

            elif args.method == "random":
                weights_t = torch.randn_like(weights_t, device=device)

            elif args.method == "random_batch_level":
                weights_t = torch.rand(1, device=device)

            # ------------------------------- Global Update ------------------------------ #
            outputs = model(inputs)
            meta_ce_loss = criterion(outputs, labels)
            weights_t1 = softmax_normalization(weights_t)
            weights_t1 = weights_t

            l_task = (weights_t1 * meta_ce_loss).mean()
            optimizer.zero_grad()
            l_task.backward()
            optimizer.step()

            # store performance
            total_loss += l_task.item()
            _, predicted = outputs.max(1)
            probs = F.softmax(outputs, dim=1).squeeze().cpu().detach().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            train_accuracy = accuracy_score(all_labels, all_predictions)
            # train_f1_score = f1_score(all_labels, all_predictions)
            print(
                f"Epoch {epoch+1}/{epochs}, Task Loss: {total_loss/(i+1)}, Fair Loss: {fair_loss}, Training Acc: {train_accuracy}"
            )

        # ----------------------------------- Eval ----------------------------------- #
        train_accuracy = accuracy_score(all_labels, all_predictions)
        valid_stats = evaluate(
            model, valid_dataloader, criterion, device, attr_ids, target_id
        )
        validation_res.append(valid_stats)
        if valid_stats["valid_accuracy"] > best_val_acc:
            best_model = copy.deepcopy(model)
            best_val_acc = valid_stats["valid_accuracy"]

        print(
            f"Epoch {epoch+1}/{epochs}, Valid Loss: {valid_stats['valid_loss']}, Valid Accuracy: {valid_stats['valid_accuracy']}, Valid F1 Score: {valid_stats['valid_f1_score']}\n"
        )

    # ---------------------------------------------------------------------------- #
    #                                   Test                                       #
    # ---------------------------------------------------------------------------- #
    # Extend to demographic attributes
    attr_ids.extend([20, 39])
    stats = evaluate(
        best_model,
        test_dataloader,
        criterion,
        device,
        attr_ids,
        target_id,
        dataset="test",
    )
    test_stats = {**stats}
    test_stats["method"] = f"{args.method}-tau{args.tau}-alpha{args.alpha}"

    print(test_stats)
    # load results to csv
    # result_df = pd.DataFrame([test_stats], columns=test_stats.keys()).set_index(
    #     "method"
    # )
    # results_path = f"../results/results-{args.task}-{attr_ids}.csv"
    # if os.path.exists(results_path):
    #     result_df.to_csv(results_path, mode="a", header=False)
    # else:
    #     result_df.to_csv(results_path)

    # if args.save_model:
    #     torch.save(
    #         best_model.state_dict(),
    #         f"../saved_models/{attr_ids}{args.model}-{args.method}-temp{args.tau}-{args.task}-seed{args.seed}.pth",
    #     )

    with open(f"./results-{args.dataset}.log", "a") as f:
        f.write(
            f"==================== {datetime.datetime.now()} ===================\n \n"
        )
        f.write(f"val_acc: {valid_stats['valid_accuracy']}\n")
        f.write(f"Detailed test performance: {test_stats}\n")
        f.write(f"args: {args}\n \n")

    print("best model saved! -- criteria: validation set accuracy")


if __name__ == "__main__":
    # Fix random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(args.device)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True

    print(f"Using PyTorch version: {torch.__version__}")
    print(f"Using CUDA version: {torch.version.cuda}")
    print(f"Using device: {args.device}")
    print(f"Using seed: {args.seed}")
    train(args)
