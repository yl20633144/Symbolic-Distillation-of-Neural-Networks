# Symbolic Distillation of Neural Networks

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## Description
This project is the submission of the Data Analysis Project as part of the MPhil in Data Intensive Science at the University of Cambridge. The primary objective of this project is to reproduce the results presented in [Discovering Symbolic Models from Deep Learning with Inductive Biases](https://arxiv.org/abs/2006.11287), and its corresponding implementation [(GitHub repository)](https://github.com/MilesCranmer/symbolic_deep_learning). 


## Data
The data for this work can be simulated from the **`simulate.py`** script, which can be found in the folder **`/src`**.

## Setup
### Repository
**Clone the Repository:**

Clone the repository to your machine with the following command:

```
git clone https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/assessments/projects/yl2063
```


**Navigate to the Project Directory:**

Navigate to the project directory with the following command:

```
cd /path/to/yl2063
```

    
### Conda environment

There is a conda environment file for the required packages and libraries for this project. You can create a conda environment from this file with:

```bash 
conda env create -n <name> --file environment_project.yml
```

### Activate the environment:

```bash 
conda activate <name>
```

## License
This project is licensed under the MIT License.


## Usage
### Python scripts:
All Python scripts are located in **`/src`**:

1. **`models.py`**  
   - Defines the graph connectivity for N-body systems.  
   - Implements four GNN variants: Standard, Bottleneck, ℓ₁-regularized, and KL-regularized.

2. **`message_processing.py`**  
   - Extracts edge-message activations from a trained GNN on the test set.  
   - Performs linear regression of message channels against ground-truth force components.  
   - Includes an extension function for one-to-one linear mapping between a message channel and a force component.

3. **`simulate.py`**  
   - Adapted from the original [implementation](https://github.com/MilesCranmer/symbolic_deep_learning).  
   - Generates trajectories for 2D many-body systems.  


### Python notebooks:
All Python notebooks are placed inside **`/src`**:

- **`spring.ipynb`**  
  Simulates the mass–spring system and train the four GNN variants. Performs linear regression on the top‐variance message channels to Hooke’s law \(F = k(r - r_0)\) and then uses PySR to extract the closed‐form expression.

- **`r2.ipynb`**  
  Implements the inverse‐square gravitational system (\(F = G\,m_i m_j / r^2\)). The rest working flow is the same as **`spring.ipynb`**.

- **`coulomb.ipynb`**  
  Studies the Coulomb interaction (\(F = q_i q_j / r^2\)). The rest working flow is the same as **`spring.ipynb`**.

- **`inverse_distance.ipynb`**  
  Covers the “toy” inverse‐distance system (\(F = k / r\)). The rest working flow is the same as **`spring.ipynb`**.

### Model weights  
The **`/model_weights`** folder contains the learned parameters for all sixteen GNN models (4 physical systems × 4 variant types).


## Report and summary
The final project report and the executive summary are located in **`/report`**.



## Notes:
1. All Random seeds were set to 0 for reproducibility.
2. Wandb visualization is implemented inside codes. To re-run the codes, please login to your own wandb account.
3. If you want to rerun the analysis or inference without retraining all models, you can load the saved weights from the **`/model_weights`** folder.  Below is an example for the ℓ₁-regularized model on one of the systems. Note that you need to make sure that you have run the codes for data generation and variable definition in the notebooks before loading model parameters.
```
import os
model_l1 = NbodyGNN(n_f, msg_dim, dim,  hidden=hidden, edge_index=get_edge_index(n, sim), aggr=aggr,l1_reg_weight=0.02).to(device)
root_checkpoints = os.path.join("..", "model_weights")
os.makedirs(root_checkpoints, exist_ok=True)

save_path = os.path.join(root_checkpoints, "r2_l1.pth")
state_dict = torch.load(save_path, map_location=device)
model_l1.load_state_dict(state_dict)


model_l1.eval()
```


## AI generation tools:
Declaration of AI generation tools, ChatGPT, for the coding part is made here:
1. It helps me format my code comments and docstrings, making the codes reader-friendly.
2. It helps me implement the graph connectivity, GNN model architechture, message extraction and data analysis pipeline.
3. It helps me make plots with publication quality.
4. It helps me debug problems in my codes.
5. It helps me format the README.md, pyproject.toml and autodocumentation.