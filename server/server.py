import socket
import time
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv

from utils.adaptive_tau import ControlAlgAdaptiveTauServer
from data.data_reader import get_data
from utils.statistics import CollectStatistics
from utils.helpers import send_msg, recv_msg
from models.mobilenet import ModelMobileNet
from config import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # GPU

model = ModelMobileNet()
if hasattr(model, 'create_graph'):
    model.create_graph(learning_rate=step_size)

if time_gen is not None:
    use_fixed_averaging_slots = True
else:
    use_fixed_averaging_slots = False

if batch_size < total_data:
    train_image, train_label, test_image, test_label, train_label_orig = get_data()

listening_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
listening_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
listening_sock.bind((SERVER_ADDR, SERVER_PORT))
client_sock_all = []

# Establish connections
while len(client_sock_all) < n_nodes:
    listening_sock.listen(n_nodes)
    print("Waiting for incoming connections...")
    (client_sock, (ip, port)) = listening_sock.accept()
    print('Got connection from ', (ip, port))
    client_sock_all.append(client_sock)

stat = CollectStatistics(results_file_name=multi_run_results_file_path, is_single_run=False)

for sim in sim_runs:
    for case in case_range:
        for tau_setup in tau_setup_all:
            csv_filename = f"CCCAFL_MobileNet_records_sim_{sim}_case_{case}_tau_{tau_setup}.csv"
            with open(csv_filename, mode='w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["global_iteration", "node_id", "tau_actual", "model_size_bytes",
                                 "power_diff_W", "cpu_diff_%", "mem_diff_MB",
                                 "bytes_sent", "bytes_recv", "local_time_s"])

                global_round_index = 0
                stat.init_stat_new_global_round()

                dim_w = model.get_weight_dimension()
                w_global_init = model.get_init_weight(dim_w, rand_seed=sim)
                w_global = w_global_init

                w_global_min_loss = None
                loss_min = np.inf
                prev_loss_is_min = False

                if tau_setup < 0:
                    is_adapt_local = True
                    tau_config = 1
                else:
                    is_adapt_local = False
                    tau_config = tau_setup

                if is_adapt_local or estimate_beta_delta_in_all_runs:
                    if tau_setup == -1:
                        control_alg = ControlAlgAdaptiveTauServer(
                            is_adapt_local, dim_w, client_sock_all, n_nodes,
                            control_param_phi, moving_average_holding_param
                        )
                    else:
                        raise Exception('Invalid setup of tau.')
                else:
                    control_alg = None

                for n in range(0, n_nodes):
                    msg = ['MSG_INIT_SERVER_TO_CLIENT',
                           num_iterations_with_same_minibatch_for_tau_equals_one,
                           step_size, control_alg, read_all_data_for_stochastic,
                           use_min_loss, sim]
                    send_msg(client_sock_all[n], msg)

                print('All clients connected')

                for n in range(0, n_nodes):
                    recv_msg(client_sock_all[n], 'MSG_DATA_PREP_FINISHED_CLIENT_TO_SERVER')

                print('Start learning')

                total_time = 0
                total_time_recomputed = 0
                it_each_local = None
                it_each_global = None

                is_last_round = False
                is_eval_only = False
                tau_new_resume = None
                opt_c = 1

                while True:
                    print('-' * 75)
                    print(f'Current tau config: {tau_config}')

                    time_total_all_start = time.time()

                    for n in range(0, n_nodes):
                        msg = ['MSG_WEIGHT_TAU_SERVER_TO_CLIENT',
                               w_global, tau_config, is_last_round,
                               prev_loss_is_min, opt_c]
                        send_msg(client_sock_all[n], msg)

                    w_global_prev = w_global
                    print('Waiting for local iteration at client')

                    w_global = np.zeros(dim_w)
                    loss_last_global = 0.0
                    loss_w_prev_min_loss = 0.0
                    received_loss_local_w_prev_min_loss = False
                    data_size_total = 0
                    time_all_local_all = 0
                    data_size_local_all = []
                    unit_resources_all = []

                    tau_actual = 0

                    for n in range(0, n_nodes):
                        msg = recv_msg(client_sock_all[n], 'MSG_WEIGHT_TIME_SIZE_CLIENT_TO_SERVER')
                        
                        w_local = msg[1]
                        time_all_local = msg[2]
                        tau_actual_tmp = msg[3]
                        data_size_local = msg[4]
                        loss_local_last_global = msg[5]
                        loss_local_w_prev_min_loss = msg[6]
                        unit_resources = msg[7]
                        mingyi_tau = msg[8]
                        approx_power_diff = msg[9]
                        cpu_usage_diff = msg[10]
                        mem_used_diff = msg[11]
                        bytes_sent_diff = msg[12]
                        bytes_recv_diff = msg[13]

                        model_size = w_local.nbytes

                        print(f"[Round {global_round_index}] Client {n}: "
                              f"tau={mingyi_tau}, size={model_size}B, "
                              f"power={approx_power_diff:.2f}W")

                        writer.writerow([global_round_index, n, mingyi_tau, model_size,
                                         approx_power_diff, cpu_usage_diff, mem_used_diff,
                                         bytes_sent_diff, bytes_recv_diff, time_all_local])
                        csvfile.flush()

                        tau_actual = mingyi_tau
                        w_global += w_local * data_size_local
                        data_size_local_all.append(data_size_local)
                        data_size_total += data_size_local
                        unit_resources_all.append(unit_resources)
                        time_all_local_all = max(time_all_local_all, time_all_local)

                        if use_min_loss:
                            loss_last_global += loss_local_last_global * data_size_local
                            if loss_local_w_prev_min_loss is not None:
                                loss_w_prev_min_loss += loss_local_w_prev_min_loss * data_size_local
                                received_loss_local_w_prev_min_loss = True

                    w_global /= data_size_total

                    # Compute optimal c for heterogeneous resources
                    def isPrime(x):
                        for i in range(2, int(x ** 0.5 + 1)):
                            if x % i == 0:
                                return False
                        return True

                    def min_gongbei(a, b):
                        if max(a, b) % min(a, b) == 0:
                            return max(a, b)
                        elif isPrime(a) and isPrime(b):
                            return a * b
                        else:
                            t = 2
                            while True:
                                if (max(a, b) * t) % (min(a, b)) == 0:
                                    break
                                t += 1
                            return max(a, b) * t

                    to_one_unit_res = []
                    am = []
                    cm = []

                    sma = min(unit_resources_all) if len(unit_resources_all) > 0 else 1
                    for i in range(len(unit_resources_all)):
                        to_one_unit_res.append(unit_resources_all[i] / sma)

                    temp = 1
                    for k in range(len(to_one_unit_res)):
                        temp = min_gongbei(temp, to_one_unit_res[k])

                    for i in range(len(to_one_unit_res)):
                        am.append(round(temp / to_one_unit_res[i]))

                    for i in range(len(unit_resources_all)):
                        cm.append(am[i] * unit_resources_all[i])

                    opt_c = max(cm) if len(cm) > 0 else 1

                    if True in np.isnan(w_global):
                        print('*** w_global is NaN, using previous value')
                        w_global = w_global_prev
                        use_w_global_prev_due_to_nan = True
                    else:
                        use_w_global_prev_due_to_nan = False

                    if use_min_loss:
                        loss_last_global /= data_size_total
                        if received_loss_local_w_prev_min_loss:
                            loss_w_prev_min_loss /= data_size_total
                            loss_min = loss_w_prev_min_loss
                        if loss_last_global < loss_min:
                            loss_min = loss_last_global
                            w_global_min_loss = w_global_prev
                            prev_loss_is_min = True
                        else:
                            prev_loss_is_min = False

                        print(f"Loss (prev global): {loss_last_global:.6f}")
                        print(f"Minimum loss: {loss_min:.6f}")

                    if not use_w_global_prev_due_to_nan:
                        if control_alg is not None:
                            tau_new = control_alg.compute_new_tau(
                                data_size_local_all, data_size_total,
                                it_each_local, it_each_global,
                                max_time, step_size, tau_config, use_min_loss
                            )
                        else:
                            tau_new = tau_new_resume if tau_new_resume is not None else tau_config
                            tau_new_resume = None
                    else:
                        if tau_new_resume is None:
                            tau_new_resume = tau_config
                        if control_alg is not None:
                            tau_new = control_alg.compute_new_tau(
                                data_size_local_all, data_size_total,
                                it_each_local, it_each_global,
                                max_time, step_size, tau_config, use_min_loss
                            )
                        tau_new = 1

                    time_total_all_end = time.time()
                    time_total_all = time_total_all_end - time_total_all_start
                    time_global_aggregation_all = max(0.0, time_total_all - time_all_local_all)

                    if use_fixed_averaging_slots:
                        if isinstance(time_gen, (list,)):
                            t_g = time_gen[case]
                        else:
                            t_g = time_gen
                        it_each_local = max(0.00000001, np.sum(t_g.get_local(tau_actual)) / tau_actual)
                        it_each_global = t_g.get_global(1)[0]
                    else:
                        it_each_local = max(0.00000001, opt_c)
                        it_each_global = time_global_aggregation_all

                    total_time_recomputed += it_each_local * tau_actual + it_each_global
                    total_time += time_total_all

                    stat.collect_stat_end_local_round(case, tau_actual, it_each_local,
                                                       it_each_global, control_alg)

                    acc = model.accuracy(test_image, test_label, w_global)
                    print(f'*** Test accuracy: {acc:.4f}')
                    epoch_loss = model.loss(train_image, train_label, w_global)
                    stat.collect_loss_and_resource(epoch_loss, total_time_recomputed, tau_actual, acc)

                    if use_min_loss:
                        tmp_time = total_time_recomputed + it_each_local * (tau_new + 1) + it_each_global * 2
                    else:
                        tmp_time = total_time_recomputed + it_each_local * tau_new + it_each_global

                    if tmp_time < max_time:
                        tau_config = tau_new
                    else:
                        if use_min_loss:
                            tau_config = int((max_time - total_time_recomputed
                                              - 2 * it_each_global - it_each_local) / it_each_local)
                        else:
                            tau_config = int((max_time - total_time_recomputed
                                              - it_each_global) / it_each_local)
                        
                        tau_config = max(1, min(tau_config, tau_new))
                        is_last_round_tmp = True

                    if is_last_round:
                        break
                    if is_eval_only:
                        tau_config = 1
                        is_last_round = True
                    if 'is_last_round_tmp' in locals() and is_last_round_tmp:
                        if use_min_loss:
                            is_eval_only = True
                        else:
                            is_last_round = True

                    global_round_index += 1

                w_eval = w_global_min_loss if (use_min_loss and w_global_min_loss is not None) else w_global

                stat.collect_stat_end_global_round(sim, case, tau_setup, total_time,
                                                   model, train_image, train_label,
                                                   test_image, test_label, w_eval,
                                                   total_time_recomputed)
