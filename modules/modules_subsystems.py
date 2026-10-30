import math
import torch
from itertools import chain
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import patches as patches
import pandas as pd
import numpy as np
import time
from scipy.stats import truncnorm
from browser.options_base import DEVICETYPE
from datasets.utils import noisify, get_concat_indices

class Subsystem:

    def __init__(self, args=None):
        self.args = args
        self.PS_list = []
        self.Devices_list = []
        self.acc_list = []
        self.ortho_list = []
        self.noisy_list = []
        self.corr_list = []
        self.corr_matrix = None
        self.corr_matrix_list = []
        self.mse_list = []
        self.cos_list = []
        self.r_list = []
        self.r_hat_list = []
        self.power_list = []
        self.sin_list = []
        self.uavloc_list = []
        self.Devices_noisy_level = []
        self.correct_total = 0

    def Subsystem_Init(self):
        for device in self.Devices_list:
            device.LocalNetwork.load_state_dict(self.PS_list[0].GlobalNetwork.state_dict())
        self.Devices_noisy_level = np.zeros(len(self.Devices_list))

    def Subsystem_DeviceDatasetNoisify(self):
        rng = np.random.default_rng(self.args.Dataseed)
        noise_or_not_client = rng.binomial(1, self.args.noise_client, len(self.Devices_list))
        random_initial = rng.random(len(self.Devices_list))
        if self.args.noise_type_sample == 'lb':
            random_initial = (1 - self.args.noise_sample) * random_initial + self.args.noise_sample
        elif self.args.noise_type_sample == 'mid':
            if self.args.noise_sample <= 0.5:
                random_initial = 2 * self.args.noise_sample * random_initial
            else:
                random_initial = (2 - 2 * self.args.noise_sample) * random_initial + (2 * self.args.noise_sample - 1)
        elif self.args.noise_type_sample == 'Gaussian':
            a = -self.args.noise_sample
            b = 1 - self.args.noise_sample
            loc = self.args.noise_sample
            scale = 1
            trunc_samples = truncnorm.rvs(a, b, loc=loc, scale=scale, size=len(self.Devices_list), random_state=rng)
            random_initial = trunc_samples
            print(random_initial)
            print(np.mean(random_initial))
        noise_rate_client = noise_or_not_client * random_initial
        for (i, device) in enumerate(self.Devices_list):
            device.LocalNetwork.data.train_labels = np.asarray([device.LocalNetwork.data.dataset.train_labels[j] for j in device.LocalNetwork.data.indices])
            if noise_or_not_client[i] > 0:
                classnum = np.array(list(set(np.unique(device.LocalNetwork.data.train_labels))))
                (device.LocalNetwork.data.train_noisy_labels, device.LocalNetwork.data.actual_noise_rate) = noisify(dataset=device.LocalNetwork.data.dataset, train_labels=device.LocalNetwork.data.train_labels, noise_type=self.args.noise_type, noise_rate=noise_rate_client[i], nb_classes=classnum, rng=rng)
                device.LocalNetwork.data.dataset.noise_or_not = np.transpose(device.LocalNetwork.data.train_noisy_labels) == np.transpose(device.LocalNetwork.data.train_labels)
                noisy_labels_temp = np.array(device.LocalNetwork.data.dataset.train_noisy_labels)
                noisy_labels_temp[device.LocalNetwork.data.indices] = device.LocalNetwork.data.train_noisy_labels
                device.LocalNetwork.data.dataset.train_noisy_labels = noisy_labels_temp
            else:
                device.LocalNetwork.data.actual_noise_rate = 0
                device.LocalNetwork.data.train_noisy_labels = device.LocalNetwork.data.train_labels
            print('Client %d, Noisy level %.2f' % (i, device.LocalNetwork.data.actual_noise_rate))
            noise_rate_client[i] = device.LocalNetwork.data.actual_noise_rate
        data_num_client = [len(device.LocalNetwork.data.train_labels) for device in self.Devices_list]
        weighted_avg_noise = np.sum(np.array(noise_rate_client) * np.array(data_num_client)) / np.sum(data_num_client)
        print('Average noise (real) %.2f' % weighted_avg_noise)

    def Subsystem_DeviceDatasetIter(self, subsystemNumber, x_slct):
        for (i, device) in enumerate(self.Devices_list):
            data_size = len(device.LocalNetwork.data)
            remainder = data_size % device.LocalNetwork.bs
            drop_last = remainder == 1
            device.LocalNetwork.iter_data = DataLoader(device.LocalNetwork.data, batch_size=device.LocalNetwork.bs, shuffle=True, drop_last=drop_last)
            device.LocalNetwork.list_num_classes = [0] * self.args.classnum
            for i in device.LocalNetwork.data.indices:
                label = device.LocalNetwork.data.dataset.train_noisy_labels[i]
                device.LocalNetwork.list_num_classes[label] += 1
        batchsize = self.args.batchsize
        if self.args.globalclassifier:
            for (i, ps) in enumerate(self.PS_list):
                ps.GlobalNetwork.traindataloader = DataLoader(ps.GlobalNetwork.traindata, batch_size=batchsize, shuffle=True, num_workers=self.args.num_workers)

    def Subsystem_FeatureCentroidInit(self):
        if self.args.loss == 'MCR2_aug_sample':
            for device in self.Devices_list:
                (Z, y, _) = device.LocalNetwork.featureextractor(device.LocalNetwork.iter_data)
                for j in y.unique():
                    device.LocalNetwork.ZZ[j] = Z[y == int(j)].T @ Z[y == int(j)]
                    device.LocalNetwork.Nj[j] = (y == int(j)).sum()
        elif self.args.loss == 'MCR2_Cov_global' or self.args.loss == 'MCR2_Cov_global_CE' or self.args.loss == 'MCR2mean':
            for device in self.Devices_list:
                device.LocalNetwork.ZZ = [0] * self.args.classnum
        elif self.args.loss == 'MCR2_centroid':
            pass

    def Subsystem_DeviceTrain(self, x_slct):
        for (i, device) in enumerate(self.Devices_list):
            if x_slct[0][i] == 1.0:
                device.LocalNetwork.start_IterTrain()
                device.LocalNetwork.start_GetGrad()

    def Subsystem_DeviceClassifier(self, x_slct):
        for (i, device) in enumerate(self.Devices_list):
            if x_slct[0][i] == 1.0:
                if not self.args.globalclassifier:
                    device.LocalNetwork.classifier0(device.LocalNetwork.iter_data)

    def Subsystem_PSTrain(self, task_id, epoch, optimizer, J):
        for (i, ps) in enumerate(self.PS_list):
            ps.GlobalNetwork.grad_update = 0
            for (j, device) in enumerate(self.Devices_list):
                if optimizer.x_slct[0][j] == 1.0:
                    ps.GlobalNetwork.grad_update += optimizer.x_slct[task_id][j] * torch.tensor(self.args.datasets_num_dvc[task_id][j]).to(DEVICETYPE) * device.LocalNetwork.grad_basic
            denominator = self.args.datasets_num[i]
            if getattr(self.args, 'participation_rates', None):
                denominator = np.dot(optimizer.x_slct[task_id], self.args.datasets_num_dvc[task_id])
            ps.GlobalNetwork.grad_update /= denominator
            ps.GlobalNetwork.grad_update = ps.GlobalNetwork.grad_update.view(-1)
            ps.GlobalNetwork.RefreshLearningRate(task_id, epoch)

    def Subsystem_ClassifierTrain(self, task_id, epoch, optimizer, J):
        if self.args.globalclassifier:
            for (i, ps) in enumerate(self.PS_list):
                ps.GlobalNetwork.classifier0(ps.GlobalNetwork.traindataloader)
        else:
            for (i, ps) in enumerate(self.PS_list):
                initial = np.zeros(self.args.classnum)
                selected_devices = []
                selected_indices = []
                global_num_classes = np.zeros(self.args.classnum)
                for (j, device) in enumerate(self.Devices_list):
                    if optimizer.x_slct[task_id][j] == 1:
                        selected_devices.append(device)
                        selected_indices.append(j)
                        global_num_classes += np.array(device.LocalNetwork.list_num_classes)
                for (j, device) in enumerate(selected_devices):
                    device.LocalNetwork.prop_class = [0] * self.args.classnum
                    for c in range(self.args.classnum):
                        if global_num_classes[c] > 0:
                            device.LocalNetwork.prop_class[c] = device.LocalNetwork.list_num_classes[c] / global_num_classes[c]
                    device.LocalNetwork.prop_class = torch.tensor(device.LocalNetwork.prop_class, dtype=torch.float32, device=DEVICETYPE)
                for c in range(self.args.classnum):
                    for (j, device) in zip(selected_indices, selected_devices):
                        if device.LocalNetwork.label_presence[c] == 0:
                            continue
                        if initial[c] == 0:
                            initial[c] = 1
                            ps.GlobalNetwork.means[c] = optimizer.x_slct[task_id][j] * device.LocalNetwork.prop_class[c] * device.LocalNetwork.means[c]
                            if self.args.align:
                                ps.GlobalNetwork.covariances[c] = optimizer.x_slct[task_id][j] * device.LocalNetwork.prop_class[c] * device.LocalNetwork.covariances[c]
                            else:
                                ps.GlobalNetwork.covariances[c] = optimizer.x_slct[task_id][j] * device.LocalNetwork.covariances[c]
                            ps.GlobalNetwork.weights[c] = optimizer.x_slct[task_id][j] * torch.tensor(self.args.datasets_num_dvc[task_id][j]).to(DEVICETYPE) / self.args.datasets_num[i] * device.LocalNetwork.weights[c]
                            ps.GlobalNetwork.subspace[c] = optimizer.x_slct[task_id][j] * device.LocalNetwork.prop_class[c] * device.LocalNetwork.subspace[c]
                        else:
                            initial[c] = initial[c] + 1
                            ps.GlobalNetwork.means[c] += optimizer.x_slct[task_id][j] * device.LocalNetwork.prop_class[c] * device.LocalNetwork.means[c]
                            if self.args.align:
                                ps.GlobalNetwork.covariances[c] += optimizer.x_slct[task_id][j] * device.LocalNetwork.prop_class[c] * device.LocalNetwork.covariances[c]
                            else:
                                ps.GlobalNetwork.covariances[c] += optimizer.x_slct[task_id][j] * device.LocalNetwork.covariances[c]
                            ps.GlobalNetwork.weights[c] += optimizer.x_slct[task_id][j] * torch.tensor(self.args.datasets_num_dvc[task_id][j]).to(DEVICETYPE) / self.args.datasets_num[i] * device.LocalNetwork.weights[c]
                            ps.GlobalNetwork.subspace[c] += optimizer.x_slct[task_id][j] * device.LocalNetwork.prop_class[c] * device.LocalNetwork.subspace[c]
                    if not self.args.align:
                        ps.GlobalNetwork.covariances[c] = ps.GlobalNetwork.covariances[c] / initial[c]
        if self.args.loss == 'MCR2_aug_sample':
            for (i, ps) in enumerate(self.PS_list):
                for c in range(self.args.classnum):
                    ps.GlobalNetwork.Nj[c] = 0
                    for (j, device) in enumerate(self.Devices_list):
                        ps.GlobalNetwork.Nj[c] += device.LocalNetwork.list_num_classes[c]
                    ps.GlobalNetwork.ZZ[c] = ps.GlobalNetwork.covariances[c] * ps.GlobalNetwork.Nj[c]
                for (m, device) in enumerate(self.Devices_list):
                    device.LocalNetwork.ZZ = ps.GlobalNetwork.ZZ
                    device.LocalNetwork.Nj = ps.GlobalNetwork.Nj
        elif self.args.loss == 'MCR2_Cov_global' or self.args.loss == 'MCR2':
            for (i, ps) in enumerate(self.PS_list):
                if set(ps.GlobalNetwork.ZZ) != {None}:
                    ps.GlobalNetwork.ZZ = [self.args.Cov_momentum * z + (1 - self.args.Cov_momentum) * c for (z, c) in zip(ps.GlobalNetwork.ZZ, ps.GlobalNetwork.covariances)]
                else:
                    ps.GlobalNetwork.ZZ = ps.GlobalNetwork.covariances
                if self.args.align:
                    for (m, device) in enumerate(self.Devices_list):
                        device.LocalNetwork.ZZ = ps.GlobalNetwork.ZZ
                        device.LocalNetwork.means = ps.GlobalNetwork.means
                        device.LocalNetwork.covariances = ps.GlobalNetwork.covariances
                        device.LocalNetwork.weights = ps.GlobalNetwork.weights
                        device.LocalNetwork.subspace = ps.GlobalNetwork.subspace
        elif self.args.loss == 'MCR2_Cov_global_CE' or self.args.loss == 'MCR2mean':
            for (i, ps) in enumerate(self.PS_list):
                if set(ps.GlobalNetwork.ZZ) != {None}:
                    ps.GlobalNetwork.ZZ = [self.args.Cov_momentum * z + (1 - self.args.Cov_momentum) * c for (z, c) in zip(ps.GlobalNetwork.ZZ, ps.GlobalNetwork.covariances)]
                else:
                    ps.GlobalNetwork.ZZ = ps.GlobalNetwork.covariances
                if self.args.align:
                    for (m, device) in enumerate(self.Devices_list):
                        device.LocalNetwork.ZZ = ps.GlobalNetwork.ZZ
                        device.LocalNetwork.means = ps.GlobalNetwork.means
                        device.LocalNetwork.covariances = ps.GlobalNetwork.covariances
                        device.LocalNetwork.weights = ps.GlobalNetwork.weights
                        device.LocalNetwork.subspace = ps.GlobalNetwork.subspace
        elif self.args.loss == 'MCR2_centroid':
            for (i, ps) in enumerate(self.PS_list):
                ps.GlobalNetwork.ZZ = ps.GlobalNetwork.means
                for (m, device) in enumerate(self.Devices_list):
                    device.LocalNetwork.ZZ = ps.GlobalNetwork.ZZ
        elif self.args.loss == 'MCR2_CE':
            for (i, ps) in enumerate(self.PS_list):
                for (m, device) in enumerate(self.Devices_list):
                    device.LocalNetwork.means = ps.GlobalNetwork.means
                    device.LocalNetwork.covariances = ps.GlobalNetwork.covariances
                    device.LocalNetwork.weights = ps.GlobalNetwork.weights

    def Subsystem_NoiseDetect(self, task_id, epoch, optimizer):
        error_counts = []
        data_num_client = []
        last_noisy_level = list(self.Devices_noisy_level)
        for (m, device) in enumerate(self.Devices_list):
            (self.Devices_noisy_level[m], correctnum) = device.LocalNetwork.detect(epoch, optimizer.x_slct[task_id][m], m, last_noisy_level)
            if self.args.correction == 1 and epoch > self.args.correct_start and (epoch < self.args.correct_end) and (epoch % self.args.correct_interval == 0):
                self.correct_total += correctnum
                print('Client %d (Select %d), Noisy level %.2f, Estimated noisy level %.2f, Correction Number: %d+%d /%d' % (m, int(optimizer.x_slct[task_id][m]), device.LocalNetwork.data.actual_noise_rate, self.Devices_noisy_level[m], device.LocalNetwork.correctnumsum, correctnum, len(device.LocalNetwork.data)))
                device.LocalNetwork.correctnumsum += correctnum
            indices = device.LocalNetwork.data.indices
            noisy_labels = np.array(device.LocalNetwork.data.dataset.train_noisy_labels)[indices]
            true_labels = np.array(device.LocalNetwork.data.dataset.train_labels)[indices]
            error_count = np.sum(noisy_labels != true_labels)
            error_counts.append(error_count)
            data_num_client.append(len(indices))
        print('Correction Number in total: {}'.format(self.correct_total))
        if epoch == 0:
            self.noisy_list.append(data_num_client)
        self.noisy_list.append(error_counts)
        global_error_rate = np.sum(error_counts) / np.sum(data_num_client)
        print('Avegrage noisy rate (real) %.4f' % global_error_rate)

    def Subsystem_DeviceRefresh(self, lr):
        for device in self.Devices_list:
            device.LocalNetwork.load_state_dict(self.PS_list[0].GlobalNetwork.state_dict(), strict=True)
