# Component-Based Fairness in Face Attribute Classification with Bayesian Network-informed Meta Learning 
Code implementation for FAccT 2025 paper "Component-Based Fairness in Face Attribute Classification with Bayesian Network-informed Meta Learning"
![Overall pipeline of BNMR](./save/pipeline.png)


# Requirement
To prepare the environment, use `./environment.txt` file and conda.

# Training
The training entry is in `./src/train.sh`. In `train.sh`, replace the data_root argument with the path to folder containing CelebA dataset. 

# Visualizations and Models 
Please find the learned Bayesian Network under `BNMR/save/(test)bnn[1, 6, 7, 14, 24].xml`, accompanied with the visualization for 5 attributes and all attributes DAG.

# Acknowledgement

During the implementation we base our code mostly on Transformers from HuggingFace and [Meta-Weight-Net](https://github.com/xjtushujun/meta-weight-net.git) Many thanks to the authors for their great work!


# Cite
Please consider citing the following papers if you use our methods/code in your research:

