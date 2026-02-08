# Coordinating Computational Capacity for Adaptive Federated Learning in Heterogeneous Edge Computing Systems
A lightweight federated learning framework for distributed machine learning on resource-constrained devices.

---

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt

```

### 2. Prepare Dataset

Create directory and place data:

```text
datasets/plant-leaf-disease-224x224/
  ├── train/
  │   └── train_data.bin        # Training data (1750 samples)
  └── test/
      └── test_data.bin         # Test data (500 samples)

```

**Data format:** 1 byte label + 150,528 bytes image (224×224×3 RGB)

### 3. Configure

Edit `config.py`:

```python
SERVER_ADDR = '192.168.x.x'  # Server IP
SERVER_PORT = 51000
n_nodes = 2                   # Number of clients
batch_size = 5
step_size = 0.001
max_time = 14400

```

---

## Run

**Terminal 1 - Server**:

```bash
python server/server.py

```

**Terminal 2, 3, ... - Clients**:

```bash
python client/client.py

```

> **Note**: Server waits for `n_nodes` clients to connect. Once all connected, federated learning starts.

---

## Output

Results saved in CSV:

* `CCCAFL_MobileNet_records_sim_*_case_*_tau_*.csv` - Per-iteration metrics
* `results/MultipleRuns.csv` - Summary statistics

---

## Citation

If you use this code or methodology in your research, please cite the following paper:

```bibtex
@article{yang2025Coordinating,
  author={Yang, Kechang and Hu, Biao and Zhao, Mingguo},
  journal={IEEE Transactions on Parallel and Distributed Systems}, 
  title={Coordinating Computational Capacity for Adaptive Federated Learning in Heterogeneous Edge Computing Systems}, 
  year={2025},
  volume={36},
  number={8},
  pages={1509-1523},
  keywords={Computational modeling;Training;Servers;Adaptation models;Internet of Things;Federated learning;Edge computing;Convergence;Heuristic algorithms;Performance evaluation;Federated learning (FL);Internet of Things;distributed gradient descent},
  doi={10.1109/TPDS.2025.3574718}
}

```

