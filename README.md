# Symbolic Distillation of Neural Networks

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## Description
This project is the submission of the Data Analysis Project as part of the MPhil in Data Intensive Science (Dissertation) at the University of Cambridge. The primary objective of this project has two parts:

1. To reproduce the results presented in [Discovering Symbolic Models from Deep Learning with Inductive Biases](https://arxiv.org/abs/2006.11287), and its corresponding implementation [(GitHub repository)](https://github.com/MilesCranmer/symbolic_deep_learning). This mainly involves recovering analytical force laws from learned message representations by combining Graph Neural Networks (GNNs) with symbolic regression.

2. Extension — We conduct an in-depth study of how message channels encode physical information. In particular, we investigate whether the edge-message space yields representations that align with true physical laws, and we illustrate that the underlying encoding mechanism is not fully transparent.

The project provides insights into the interpretability of deep neural networks, especially in scientific applications.


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
  Simulates the mass–spring system and trains the four GNN variants. Performs linear regression on the top-variance message channels against Hooke’s law \(F = k(r - r_0)\), uses PySR to extract the closed-form expression, and implements **Extension II** (channel-wise specialization).

- **`r2.ipynb`**  
  Implements the inverse-square gravitational system (\(F = G\,m_i m_j / r^2\)). Follows the same workflow as **`spring.ipynb`** and includes **Extension II**.

- **`charge.ipynb`**  
  Studies the Coulomb interaction (\(F = q_i q_j / r^2\)). Follows the same workflow as **`spring.ipynb`**.

- **`inverse_distance.ipynb`**  
  Covers the “toy” inverse-distance system (\(F = k / r\)). Follows the same workflow as **`spring.ipynb`** and implements both **Extension I** (acceleration vs. force encoding) and **Extension II** (channel-wise specialization).

### Model weights  
The **`/model_weights`** folder contains the learned parameters for all sixteen GNN models (4 physical systems × 4 variant types).

### Symbolic regression results
All symbolic regression results, including both the **core systems** and the **extension experiments**, can be found in the `src/outputs` directory.

For each model, the outputs include:

- `hall_of_fame.csv`: A table containing the top symbolic expressions discovered by the regression algorithm, along with their corresponding scores (e.g., accuracy, complexity, loss).
- `checkpoint.pkl`: A serialized PySR run containing the symbolic model’s internal state, including optimizer history.
- `hall_of_fame.csv.bak`: A backup of the results file.


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
### Auto-Generated Documentation

The full API documentation is automatically generated using [Sphinx](https://www.sphinx-doc.org/) and [AutoAPI](https://sphinx-autoapi.readthedocs.io/). It includes detailed descriptions of all Python modules and functions under the `src/` directory.

To view the documentation locally, open the `index.html` file located in the `docs/build/html/` directory with any web browser:




## AI generation tools:
Declaration of AI generation tools, ChatGPT, for the coding part is made here:
1. It helps me format my code comments and docstrings, making the codes reader-friendly.
2. It helps me implement the graph connectivity, GNN model architechture, message extraction and data analysis pipeline.
3. It helps me make plots with publication quality.
4. It helps me debug problems in my codes.
5. It helps me format the README.md, pyproject.toml and autodocumentation.