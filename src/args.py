import argparse
import pandas as pd


# fmt: off
parser = argparse.ArgumentParser(description="Training args for classification task")


# ------------------------------- Global Params ------------------------------ #
parser.add_argument("--task", type=str, required=False, default="smile", help="Name of downstream task.")
parser.add_argument("--model", type=str, required=False, default="lightCNN", help="Model for downstream task.")
parser.add_argument("--epochs", type=int, required=False, default=5, help="Number of epochs.")
parser.add_argument("--seed", type=int, required=False, default=5, help="Random seed.")
parser.add_argument("--device", type=str, required=False, default="cuda:0", help="Device for training.")
parser.add_argument("--lr", type=float, required=False, default=1e-4, help="Learning rate for main training.")
parser.add_argument("--use_subset", type=bool, required=False, default=True, help="Use only part of data for training.")
parser.add_argument("--save_model", type=bool, required=False, default=False, help="Save trained model.")
parser.add_argument("--dataset", type=str, required=False, default="celeba", help="Placeholder for new dataset.")

# ------------------------------ Dataset Related ----------------------------- #
parser.add_argument("--img_size", type=int, required=False, default=224, help="Torch tranformation image size.")
parser.add_argument("--batch_size", type=int, required=False, default=16, help="Batch size for classification.")
# ------------------------------ Method Specific ----------------------------- #
parser.add_argument("--debias", type=bool, required=False, default=True, help="Use re-weighting to debias.")
parser.add_argument("--method", type=str, required=False, default="Bayesian", help="Specify weighting scheme from [uniform, random, random_batch_level, FORML, Ablation, Bayesian_online, Bayesian_Finetune]")
parser.add_argument("--v_batch_size", type=int, required=False, default=16, help="Batch size for weight net meta task.")
parser.add_argument("--v_epochs", type=int, required=False, default=3, help="Number of epochs for re-weighting net update.")
parser.add_argument("--v_lr", type=float, required=False, default=10, help="Learning rate for weight update.")
parser.add_argument("--tau", type=float, required=False, default=0.3, help="Temperature for softmax normalization")
parser.add_argument("--subgroup_size", type=int, required=False, default=20, help="K-shot Fair evaluation")
parser.add_argument("--epoch", type=int, required=False, default=0, help="For evaluation")
parser.add_argument("--load_model", type=str, required=False, default="none", help="Load model to finetune.")
parser.add_argument("--online", type=bool, required=False, default=False, help="Update BNN during training.")
parser.add_argument("--calibration_warmstart", type=int, required=False, default=0, help="Calibration starting epoch.")
parser.add_argument("--data_root", type=str, required=True, default="", help="Data root")
parser.add_argument("--note", type=str, required=False, default="", help="Notes to keep track")


# Parse args
args = parser.parse_args()
demographics_attr = ['Attractive', 'Male', 'Young', 'Smiling']
attributes = []
target_id = 0
if args.task == "smile":
    target_id = -9
elif args.task == "attractiveness":
    target_id = 2
elif args.task == "hat":
    target_id = -5
elif args.task == "young":
    target_id = -1


attributes_df = pd.read_csv(
    args.data_root + "/celeba/list_attr_celeba.txt", delim_whitespace=True, header=1
)
columns = attributes_df.columns
attr_names = columns.tolist()
bnet_columns = attr_names + [f"Model Prediction ({args.task})"]
print(attr_names)

ids = [1, 6, 7, 14, 24]
demographics = [20, 39]
print([attr_names[i] for i in ids])
print([attr_names[i] for i in demographics])
print(args)
