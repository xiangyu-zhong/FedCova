from types import SimpleNamespace
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
import modules.operations_base as opr_base
from modules.modules_subsystems import Subsystem
from modules.modules_machine import PS, Device
from modules.modules_networks import GlobalNetwork, LocalNetwork
from browser.options_base import DEVICETYPE

class Environments:

    def __init__(self, args):
        self.args = args
        self.subsystems_list = []
        self.nmse_tofile = []
        self.optimizer = None
        self.corr_curve = None

    def Environments_Init(self, dataset):
        self.Environments_TaskDivision(dataset)
        for (i, subsystem) in enumerate(self.subsystems_list):
            subsystem.Subsystem_Init()
            if not self.args.globalnoise and self.args.noise_type != 'clean':
                subsystem.Subsystem_DeviceDatasetNoisify()
        self.optimizer = SimpleNamespace(x_slct={})

    def Environments_TaskDivision(self, dataset):
        for (i, task) in enumerate(self.args.task_list):
            subsystem_temp = Subsystem(self.args)
            if self.args.globalclassifier:
                subsystem_temp.PS_list.append(PS(self.args, GlobalNetwork(self.args, dataset[task + '_test_origin'], dataset[task + '_train_origin'])))
            else:
                subsystem_temp.PS_list.append(PS(self.args, GlobalNetwork(self.args, dataset[task + '_test_origin'])))
            for j in range(self.args.users_num[i]):
                subsystem_temp.Devices_list.append(Device(self.args, LocalNetwork(self.args, i, j, dataset[task + '_train_users'][j])))
            self.subsystems_list.append(subsystem_temp)

    def Environments_Iter(self, epoch):
        if epoch % self.args.J == 0:
            # Preserve the original simulation's NumPy RNG sequence.
            for _ in range(self.args.Q * self.args.N * sum(self.args.users_num)):
                np.random.random()
        self.optimizer = opr_base.all_choice(self.args, self.optimizer)
        for (i, subsystem) in enumerate(self.subsystems_list):
            subsystem.Subsystem_DeviceDatasetIter(i, self.optimizer.x_slct)
            if epoch == 0 and self.args.featurelearning:
                subsystem.Subsystem_FeatureCentroidInit()
                subsystem.Subsystem_DeviceClassifier(self.optimizer.x_slct)
                subsystem.Subsystem_ClassifierTrain(i, epoch, self.optimizer, self.args.J)
        for (i, subsystem) in enumerate(self.subsystems_list):
            subsystem.Subsystem_DeviceTrain(self.optimizer.x_slct)
        if self.args.method == 0:
            for (i, subsystem) in enumerate(self.subsystems_list):
                subsystem.Subsystem_PSTrain(i, epoch, self.optimizer, self.args.J)
        for (i, subsystem) in enumerate(self.subsystems_list):
            subsystem.Subsystem_DeviceRefresh(subsystem.PS_list[0].GlobalNetwork.args.lr[i])
        for (i, subsystem) in enumerate(self.subsystems_list):
            if self.args.featurelearning:
                subsystem.Subsystem_DeviceClassifier(self.optimizer.x_slct)
                subsystem.Subsystem_ClassifierTrain(i, epoch, self.optimizer, self.args.J)
                if self.args.correction == 1:
                    subsystem.Subsystem_NoiseDetect(i, epoch, self.optimizer)

    def Environments_Test(self, epoch):
        if not epoch % self.args.tp:
            for (i, subsystem) in enumerate(self.subsystems_list):
                subsystem.PS_list[0].GlobalNetwork.start_test(subsystem.PS_list[0].GlobalNetwork.data)
                print('     network%d' % i + '-accuracy:%f' % subsystem.PS_list[0].GlobalNetwork.accuracy[int(epoch / self.args.tp)])
                subsystem.acc_list.append(subsystem.PS_list[0].GlobalNetwork.accuracy[int(epoch / self.args.tp)])
                if self.args.featurelearning:
                    print('     orthogonality:%f' % subsystem.PS_list[0].GlobalNetwork.ortho[int(epoch / self.args.tp)])
                    subsystem.ortho_list.append(subsystem.PS_list[0].GlobalNetwork.ortho[int(epoch / self.args.tp)])
