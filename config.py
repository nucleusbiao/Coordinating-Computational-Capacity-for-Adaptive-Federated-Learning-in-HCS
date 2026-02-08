from time_generation import TimeGeneration
import os

# Network Configuration
SERVER_ADDR = '192.168.31.153'  # Change to your server's IP address
# SERVER_ADDR = 'localhost'  # Use for local testing
SERVER_PORT = 51000

# Results Configuration
results_file_path = os.path.join(os.path.dirname(__file__), 'results')
single_run_results_file_path = results_file_path + '/SingleRun.csv'
multi_run_results_file_path = results_file_path + '/MultipleRuns.csv'

# Control Parameters
control_param_phi = 0.00005  # Control parameter for adaptive tau algorithm

# Number of Clients
n_nodes = 1  # Specifies the total number of clients

# Moving Average Parameter
moving_average_holding_param = 0.0  # Coefficient to smooth estimation of beta, delta, and rho

# Training Parameters
step_size = 0.001  # Learning rate

# Batch Configuration
batch_size = 10  # Mini-batch size for stochastic gradient descent
total_data = 5750  # Total data samples available

# Run Configuration
single_run = False  # Set True for single run with plots, False for multiple runs

# Beta/Delta Estimation
estimate_beta_delta_in_all_runs = False  # Enable to estimate beta and delta in all runs

# Loss Strategy
use_min_loss = True  # Return weight with minimum loss (recommended for distributed case)

# Minibatch Configuration
num_iterations_with_same_minibatch_for_tau_equals_one = 3  # Reuse minibatch for efficiency

# Data Reading Strategy
read_all_data_for_stochastic = True  # Read all data into memory (faster but uses more memory)

# Case Configuration
MAX_CASE = 1  # Maximum number of cases
tau_max = 100  # Maximum value of tau

# Tau Setup
if not single_run:
    tau_setup_all = [-1]  # -1 for adaptive, positive values for fixed tau
    sim_runs = range(0, 1)  # Simulation seeds
    case_range = range(0, MAX_CASE)
else:
    case_range = [0]
    tau_setup_all = [-1]
    sim_runs = [0]

# Time Budget
max_time = 14400  # Total time budget in seconds (4 hours)

# Time Generation (None for actual measured time)
time_gen = None

# Optional: Synthetic time generation
multiply_global = 1.0
multiply_local = 1.0
