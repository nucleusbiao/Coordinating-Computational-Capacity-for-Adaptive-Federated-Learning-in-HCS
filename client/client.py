import socket
import time
import struct
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import psutil
from utils.adaptive_tau import ControlAlgAdaptiveTauClient, ControlAlgAdaptiveTauServer
from data.data_generator import Apple_leaf_data_generator
from utils.helpers import send_msg, recv_msg
from models.mobilenet import ModelMobileNet
from config import SERVER_ADDR, SERVER_PORT


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Use CPU

# CPU TDP for power estimation (adjust based on your device)
CPU_TDP_WATTS = 45.0


def approximate_cpu_power(interval=0.5, cpu_tdp=CPU_TDP_WATTS):
    """
    Estimate CPU power consumption based on CPU usage percentage.
    Assumes linear relationship between usage and power.
    """
    psutil.cpu_percent(interval=None)  # Initialize
    usage_percent = psutil.cpu_percent(interval=interval)
    return cpu_tdp * (usage_percent / 100.0)


sock = socket.socket()
sock.connect((SERVER_ADDR, SERVER_PORT))

print('---------------------------------------------------------------------------')

batch_size_prev = None
total_data_prev = None
sim_prev = None
batch_size = 5
total_data = 2250
sample_list = range(0, 1750)  # Training samples

try:
    while True:
        print('Reading all data samples used in training...')
        data_gen = Apple_leaf_data_generator(sample_list, batch_size)

        msg = recv_msg(sock, 'MSG_INIT_SERVER_TO_CLIENT')
        
        num_iterations_with_same_minibatch_for_tau_equals_one = msg[1]
        step_size = msg[2]
        control_alg_server_instance = msg[3]
        read_all_data_for_stochastic = msg[4]
        use_min_loss = msg[5]
        sim = msg[6]

        model = ModelMobileNet()
        model2 = ModelMobileNet()

        if hasattr(model, 'create_graph'):
            model.create_graph(learning_rate=step_size)
        if hasattr(model2, 'create_graph'):
            model2.create_graph(learning_rate=step_size)

        batch_size_prev = batch_size
        total_data_prev = total_data
        sim_prev = sim

        if batch_size >= total_data:
            train_indices = None
        else:
            train_indices = None
        last_batch_read_count = None

        data_size_local = len(sample_list)

        if isinstance(control_alg_server_instance, ControlAlgAdaptiveTauServer):
            control_alg = ControlAlgAdaptiveTauClient()
        else:
            control_alg = None

        w_prev_min_loss = None
        w_last_global = None
        total_iterations = 0
        learning_step_size = step_size
        unit_resources = 1

        msg = ['MSG_DATA_PREP_FINISHED_CLIENT_TO_SERVER']
        send_msg(sock, msg)

        while True:
            print('---------------------------------------------------------------------------')

            # Resource monitoring before training
            cpu_usage_before = psutil.cpu_percent(interval=None)
            cpu_usage_before = psutil.cpu_percent(interval=1.0)
            mem_info_before = psutil.virtual_memory()
            mem_used_before = mem_info_before.used
            net_before = psutil.net_io_counters()
            bytes_sent_before = net_before.bytes_sent
            bytes_recv_before = net_before.bytes_recv
            approx_power_before = approximate_cpu_power(interval=1.0)
            
            print(f"[Before] CPU={cpu_usage_before:.2f}%, "
                  f"Mem={mem_used_before / (1024 * 1024):.2f}MB, "
                  f"Power≈{approx_power_before:.2f}W")

            msg = recv_msg(sock, 'MSG_WEIGHT_TAU_SERVER_TO_CLIENT')
            w = msg[1]
            tau_config = msg[2]
            is_last_round = msg[3]
            prev_loss_is_min = msg[4]
            opt_c = msg[5]

            am = round(opt_c / unit_resources)

            if prev_loss_is_min or ((w_prev_min_loss is None) and (w_last_global is not None)):
                w_prev_min_loss = w_last_global

            if control_alg is not None:
                control_alg.init_new_round(w)

            time_local_start = time.time()

            grad = None
            loss_last_global = None
            loss_w_prev_min_loss = None
            tau_actual = 0

            # Local training iterations
            for i in range(0, am * tau_config):
                if batch_size < total_data:
                    if (not isinstance(control_alg, ControlAlgAdaptiveTauClient)) or (i != 0) or (train_indices is None) \
                            or (tau_config <= 1 and
                                (last_batch_read_count is None or
                                 last_batch_read_count >= num_iterations_with_same_minibatch_for_tau_equals_one)):

                        if read_all_data_for_stochastic:
                            train_image, train_label, actual_train_indices = next(data_gen)
                            train_indices = range(0, batch_size)
                        else:
                            train_image, train_label, _ = next(data_gen)
                            train_indices = range(0, min(batch_size, len(train_label)))

                        last_batch_read_count = 0

                    last_batch_read_count += 1

                grad = model.gradient(train_image, train_label, w, train_indices)

                if i == 0:
                    loss_last_global = model.loss(train_image, train_label, w, train_indices)
                    print('*** Loss computed from data')
                    w_last_global = w

                    if use_min_loss:
                        if (batch_size < total_data) and (w_prev_min_loss is not None):
                            loss_w_prev_min_loss = model.loss(train_image, train_label, w_prev_min_loss, train_indices)

                # Update weights using gradient descent
                w = w - learning_step_size * grad
                tau_actual += 1
                total_iterations += 1

                # Check if we should stop early (for adaptive control)
                if control_alg is not None:
                    is_last_local = control_alg.update_after_each_local(i, w, grad, total_iterations)
                    if is_last_local:
                        break

            time_local_end = time.time()
            time_all_local = time_local_end - time_local_start
            print('time_all_local =', time_all_local)

            if control_alg is not None:
                control_alg.update_after_all_local(model, train_image, train_label, train_indices,
                                                   w, w_last_global, loss_last_global)

            unit_resources = time_all_local / tau_actual

            # Resource monitoring after training
            approx_power_after = approximate_cpu_power(interval=0.1)
            cpu_usage_after = psutil.cpu_percent(interval=0.1)
            mem_info_after = psutil.virtual_memory()
            mem_used_after = mem_info_after.used
            net_after = psutil.net_io_counters()
            bytes_sent_after = net_after.bytes_sent
            bytes_recv_after = net_after.bytes_recv
            
            print(f"[After] CPU={cpu_usage_after:.2f}%, "
                  f"Mem={mem_used_after / (1024 * 1024):.2f}MB, "
                  f"Power≈{approx_power_after:.2f}W")

            cpu_usage_diff = cpu_usage_after - cpu_usage_before
            mem_used_diff = (mem_used_after - mem_used_before) / (1024 * 1024)
            bytes_sent_diff = bytes_sent_after - bytes_sent_before
            bytes_recv_diff = bytes_recv_after - bytes_recv_before
            approx_power_diff = approx_power_after - approx_power_before
            
            print(f"[Diff] CPU={cpu_usage_diff:.2f}%, Mem={mem_used_diff:.2f}MB, "
                  f"TX={bytes_sent_diff}B, RX={bytes_recv_diff}B, Power≈{approx_power_diff:.2f}W")

            msg = ['MSG_WEIGHT_TIME_SIZE_CLIENT_TO_SERVER', w, time_all_local, tau_actual, data_size_local,
                   loss_last_global, loss_w_prev_min_loss, unit_resources, tau_config,
                   approx_power_diff, cpu_usage_diff, mem_used_diff, bytes_sent_diff, bytes_recv_diff]
            send_msg(sock, msg)

            if control_alg is not None:
                control_alg.send_to_server(sock)

            if is_last_round:
                break

except (struct.error, socket.error):
    print('Server has stopped')
    pass
