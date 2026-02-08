import numpy as np
import os, sys
from adaptive_tau import ControlAlgAdaptiveTauServer
from config import *
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class CollectStatistics:
    def __init__(self, results_file_name=os.path.dirname(__file__)+'/results.csv', is_single_run=False):
        self.results_file_name = results_file_name
        self.is_single_run = is_single_run

        if not os.path.exists(os.path.dirname(results_file_name)):
            os.makedirs(os.path.dirname(results_file_name))
        if is_single_run:
            with open(results_file_name, 'a') as f:
                f.write(
                    'case,tValue,lossValue,predictionAccuracy,betaAdapt,deltaAdapt,rhoAdapt,tau,it_each_local,it_each_global\n')
                f.close()
        else:
            with open(results_file_name, 'a') as f:
                f.write(
                    'Type,Simulation,case,tau_setup,lossValue,predictionAccuracy,avg_tau,stddev_tau,' +
                    'avg_each_local,stddev_each_local,avg_each_global,stddev_each_global,' +
                    'avg_betaAdapt,stddev_betaAdapt,' +
                    'avg_deltaAdapt,stddev_deltaAdapt,avg_rhoAdapt,stddev_rhoAdapt,' +
                    'total_time_recomputed\n')
                f.close()

    def init_stat_new_global_round(self):
        if self.is_single_run:
            self.loss_values = []
            self.prediction_accuracies = []
            self.t_values = []

        self.taus = []
        self.each_locals = []
        self.each_globals = []
        self.beta_adapts = []
        self.delta_adapts = []
        self.rho_adapts = []

    def collect_stat_end_local_round(self, case, tau, it_each_local, it_each_global, control_alg):
        self.taus.append(tau)
        self.each_locals.append(it_each_local)
        self.each_globals.append(it_each_global)

        if control_alg is not None:
            if isinstance(control_alg, ControlAlgAdaptiveTauServer):
                if control_alg.beta_adapt_mvaverage is not None:
                    self.beta_adapts.append(control_alg.beta_adapt_mvaverage)
                elif self.is_single_run:
                    self.beta_adapts.append(np.nan)

                if control_alg.delta_adapt_mvaverage is not None:
                    self.delta_adapts.append(control_alg.delta_adapt_mvaverage)
                elif self.is_single_run:
                    self.delta_adapts.append(np.nan)

                if control_alg.rho_adapt_mvaverage is not None:
                    self.rho_adapts.append(control_alg.rho_adapt_mvaverage)
                elif self.is_single_run:
                    self.rho_adapts.append(np.nan)
        else:
            if self.is_single_run:
                self.beta_adapts.append(np.nan)
                self.delta_adapts.append(np.nan)
                self.rho_adapts.append(np.nan)

    def collect_loss_and_resource(self, epoch_loss, total_time_recomputed, tau_actual, acc):
        """Collect loss, accuracy and resource usage"""
        print(f"Loss: {epoch_loss}, Time: {total_time_recomputed}, Tau: {tau_actual}, Accuracy: {acc}")

    def collect_stat_end_global_round(self, sim, case, tau_setup, total_time, model, train_image, train_label,
                                      test_image, test_label, w_eval, total_time_recomputed):
        loss_final = model.loss(test_image, test_label, w_eval)
        accuracy_final = model.accuracy(test_image, test_label, w_eval)

        if not self.is_single_run:
            taus_array = np.array(self.taus)
            avg_tau = np.sum(np.dot(taus_array, taus_array)) / np.sum(taus_array)
            stddev_tau = np.std(taus_array)
            avg_each_local = np.mean(np.array(self.each_locals))
            stddev_each_local = np.std(np.array(self.each_locals))
            avg_each_global = np.mean(np.array(self.each_globals))
            stddev_each_global = np.std(np.array(self.each_globals))
            avg_beta_adapt = np.mean(np.array(self.beta_adapts))
            stddev_beta_adapt = np.std(np.array(self.beta_adapts))
            avg_delta_adapt = np.mean(np.array(self.delta_adapts))
            stddev_delta_adapt = np.std(np.array(self.delta_adapts))
            avg_rho_adapt = np.mean(np.array(self.rho_adapts))
            stddev_rho_adapt = np.std(np.array(self.rho_adapts))

            if case is None or np.isnan(case):
                case = None
                type_str = 'centralized'
            else:
                type_str = 'distributed'

            with open(self.results_file_name, 'a') as f:
                f.write(type_str + ',' + str(sim) + ',' + str(case) + ',' + str(tau_setup) + ','
                        + str(loss_final) + ',' + str(accuracy_final) + ',' + str(avg_tau) + ',' + str(stddev_tau) + ','
                        + str(avg_each_local) + ',' + str(stddev_each_local) + ','
                        + str(avg_each_global) + ',' + str(stddev_each_global) + ','
                        + str(avg_beta_adapt) + ',' + str(stddev_beta_adapt) + ','
                        + str(avg_delta_adapt) + ',' + str(stddev_delta_adapt) + ','
                        + str(avg_rho_adapt) + ',' + str(stddev_rho_adapt) + ','
                        + str(total_time_recomputed) + ','
                        + '\n')
                f.close()

        print('total time', total_time)
        print('loss value', loss_final)
        print('accuracy', accuracy_final)
