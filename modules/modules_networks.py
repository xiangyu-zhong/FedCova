import torch
from itertools import chain
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch import nn
from collections import OrderedDict
import numpy as np
import copy
import re
from torch.autograd import Variable
import torchvision.models as models
import modules.operations_base as opr_base
from browser.options_base import DEVICETYPE
import modules.loss as loss
import modules.classifier as cl
from torch.cuda.amp import autocast, GradScaler

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class BasicNetwork(torch.nn.Module):

    def __init__(self, args, data, block=BasicBlock, num_blocks=[2, 2, 2, 2], feature_dim=128, input_channel=3, num_classes=10):
        super(BasicNetwork, self).__init__()
        self.args = args
        self.data = data
        self.loss = []
        self.accuracy = []
        self.ortho = []
        self.bs = 0
        self.lr = 0
        self.v = 0
        self.mt = 0
        self.vt = 0
        self.num_classes = self.args.classnum
        self.means = [None] * self.num_classes
        self.covariances = [None] * self.num_classes
        self.weights = [None] * self.num_classes
        self.subspace = [None] * self.num_classes
        self.label_presence = np.zeros(args.num_classes, dtype=np.float32)
        self.prop_class_inner = np.zeros(args.num_classes, dtype=np.float32)
        self.ZZ = [None] * self.num_classes
        self.Nj = [0] * self.num_classes
        if self.args.network == 'CNN':
            self.conv = nn.Sequential(OrderedDict([('conv1', nn.Conv2d(1, 16, 5)), ('maxpool1', nn.MaxPool2d(2, 2)), ('relu1', nn.ReLU()), ('conv2', nn.Conv2d(16, 32, 5)), ('drop1', nn.Dropout2d()), ('maxpool2', nn.MaxPool2d(2, 2)), ('relu2', nn.ReLU())])).to(DEVICETYPE)
            self.fc = nn.Sequential(OrderedDict([('fc1', nn.Linear(32 * 4 * 4, 50)), ('relu3', nn.ReLU()), ('drop2', nn.Dropout()), ('fc2', nn.Linear(50, 10))])).to(DEVICETYPE)
            self.reduction = nn.Sequential(nn.Linear(32 * 4 * 4, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Dropout(), nn.Linear(512, self.args.featuredim)).to(DEVICETYPE)
            self.criterion = nn.CrossEntropyLoss().to(DEVICETYPE)
        elif self.args.network in ['Resnet18', 'Resnet34', 'Resnet50']:
            if self.args.network == 'Resnet34':
                block = BasicBlock
                num_blocks = [3, 4, 6, 3]
            elif self.args.network == 'Resnet50':
                block = Bottleneck
                num_blocks = [3, 4, 6, 3]
            num_classes = self.args.classnum
            self.in_planes = 64
            self.feature_dim = args.featuredim
            if self.args.network == 'Resnet50' and self.args.pretrained:
                base_model = models.resnet50(pretrained=True).to(DEVICETYPE)
                self.basemodel = nn.Sequential(*list(base_model.children())[:-2]).to(DEVICETYPE)
            else:
                self.conv1 = nn.Conv2d(input_channel, 64, kernel_size=3, stride=1, padding=1, bias=False).to(DEVICETYPE)
                self.bn1 = nn.BatchNorm2d(64).to(DEVICETYPE)
                self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1).to(DEVICETYPE)
                self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2).to(DEVICETYPE)
                self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2).to(DEVICETYPE)
                self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2).to(DEVICETYPE)
            self.reduction = nn.Sequential(nn.Linear(512 * block.expansion, 512, bias=False), nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Linear(512, self.feature_dim, bias=True)).to(DEVICETYPE)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1)).to(DEVICETYPE)
            self.fc = nn.Linear(512 * block.expansion, num_classes).to(DEVICETYPE)
            self.criterion = nn.CrossEntropyLoss().to(DEVICETYPE)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.args.network == 'CNN':
            feature = self.conv(x)
            if self.args.featurelearning:
                feature = feature.view(feature.shape[0], -1)
                feature = self.reduction(feature)
                return F.normalize(feature)
            else:
                output = self.fc(feature.view(feature.shape[0], -1))
                return output
        elif self.args.network in ['Resnet18', 'Resnet34', 'Resnet50']:
            if self.args.pretrained and self.args.network == 'Resnet50':
                out = self.basemodel(x)
            else:
                out = F.relu(self.bn1(self.conv1(x)))
                out = self.layer1(out)
                out = self.layer2(out)
                out = self.layer3(out)
                out = self.layer4(out)
            if self.args.featurelearning:
                out = nn.AdaptiveAvgPool2d(1)(out)
                out = out.view(out.size(0), -1)
                out = self.reduction(out)
                return F.normalize(out)
            else:
                out = self.avgpool(out)
                out = torch.flatten(out, 1)
                out = self.fc(out)
                return out

    def featureextractor(self, dataload):
        self.eval()
        train_features = []
        train_labels = []
        train_index = []
        with torch.no_grad():
            for (data, label, index) in dataload:
                feature = self.__call__(data.to(DEVICETYPE))
                train_features.append(feature)
                if self.args.softPi:
                    label = torch.argmax(label, dim=1)
                train_labels.append(label)
                train_index.append(index)
        train_features = torch.cat(train_features)
        train_labels = torch.cat(train_labels)
        train_index = torch.cat(train_index)
        return (train_features, train_labels, train_index)

    def classifier0(self, traindata):
        (train_features, train_labels, train_index) = self.featureextractor(traindata)
        (unique_labels, counts) = torch.unique(train_labels, return_counts=True)
        unique_labels = unique_labels.numpy()
        counts = counts.numpy()
        self.label_presence[unique_labels] = 1
        self.prop_class_inner[unique_labels] = counts / train_labels.shape[0]
        (train_acc, self.means, self.covariances, self.weights, self.subspace) = cl.evaluate('train', self.args.classifier, train_features.to(torch.float64), train_labels.to(torch.float64), num_classes=self.prop_class_inner, return_pred=False)
        self.subspace = copy.deepcopy(self.means)

    def start_test(self, data):
        self.eval()
        with torch.no_grad():
            test_data = DataLoader(data, batch_size=32, shuffle=True, num_workers=self.args.num_workers)
            accuracy = 0
            if self.args.featurelearning:
                if len(self.accuracy) > self.args.correct_start:
                    eps = 0.01
                else:
                    eps = 0.0001
                (test_features, test_labels, _) = self.featureextractor(test_data)
                (accuracy, pred) = cl.evaluate('test', self.args.classifier, test_features, test_labels, self.means, self.covariances, self.weights, self.subspace, epsilon=eps)
                self.accuracy.append(accuracy)
                orthogonality = torch.abs(torch.stack([torch.linalg.eigh(cov)[1][:, -1] for cov in self.covariances]) @ torch.stack([torch.linalg.eigh(cov)[1][:, -1] for cov in self.covariances]).T)
                mean_off_diag = orthogonality.abs()[~torch.eye(orthogonality.size(0), dtype=torch.bool, device=orthogonality.device)].mean().item()
                self.ortho.append(mean_off_diag)
            else:
                for (x_batchs, y_batchs, _) in test_data:
                    pred_batchs = self.__call__(x_batchs.to(DEVICETYPE))
                    pred_batchs = torch.topk(pred_batchs, 1)[1].squeeze(1).to(DEVICETYPE)
                    out = [i == 0 for i in pred_batchs - y_batchs.to(DEVICETYPE)]
                    accuracy += float(sum(out))
                self.accuracy.append(accuracy / len(data) * 100)
        return accuracy

class LocalNetwork(BasicNetwork):

    def __init__(self, args, task_id, device_id, data):
        super(LocalNetwork, self).__init__(args, data)
        self.iter_data = []
        self.iter_data2 = []
        self.grad_basic = torch.zeros([1, 1])
        self.grad_normal = torch.zeros([1, 1])
        self.grad_update = 0
        self.grad_mean = 0
        self.grad_std = 0
        self.Qki = self.args.datasets_num_dvc[task_id][device_id]
        self.bs = args.batchsize
        self.old_dict = None
        self.lr = args.lr[task_id]
        self.cl = args.classifier
        self.list_num_classes = [0] * args.classnum
        self.prop_class = []
        self.correctnumsum = 0

    def start_IterTrain(self):
        self.train()
        self.old_dict = copy.deepcopy(self.state_dict())
        scaler = GradScaler('cuda:0')
        if self.args.optimizer == 0:
            optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        elif self.args.optimizer == 1:
            if self.args.network in ['Resnet18', 'Resnet34', 'Resnet50']:
                optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9, weight_decay=0.0005)
            else:
                optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.5)
            if self.args.lrmom_adjust == 2:
                scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [80, 160], gamma=0.1)
        elif self.args.optimizer == 2:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, betas=[0.9, 0.999])
        for i in range(self.args.FedAvg):
            for (x_batchs, y_batchs, indexes) in self.iter_data:
                if self.args.featurelearning:
                    labels = y_batchs.to(DEVICETYPE)
                    h1 = self.__call__(x_batchs.to(DEVICETYPE))
                    self.loss = loss.loss_MCR2_Cov(self.ZZ, h1, labels)
                else:
                    self.loss = self.criterion(self.__call__(x_batchs.to(DEVICETYPE)), y_batchs.to(DEVICETYPE))
                optimizer.zero_grad()
                self.loss.backward()
                optimizer.step()
                if self.args.lrmom_adjust == 2:
                    scheduler.step()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def start_GetGrad(self):
        reshape_data_old = []
        reshape_data_new = []
        for (key, value) in self.old_dict.items():
            reshape_data_old.append(value.view(1, value.numel()))
        for (key, value) in self.state_dict().items():
            reshape_data_new.append(value.view(1, value.numel()))
        self.grad_basic = torch.cat(reshape_data_new, 1).to(DEVICETYPE) - torch.cat(reshape_data_old, 1).to(DEVICETYPE)
        self.grad_basic = torch.where(torch.isnan(self.grad_basic), torch.full_like(self.grad_basic, 0), self.grad_basic)
        if self.grad_std < 1e-05:
            self.grad_std = 1e-05
        self.grad_mean = torch.mean(self.grad_basic.cpu())
        self.grad_std = torch.std(self.grad_basic.cpu())
        grad_normal = (self.grad_basic - self.grad_mean) / self.grad_std
        self.grad_normal = opr_base.tensor2npcplx(grad_normal.squeeze(dim=0).cpu())
        self.grad_mean = np.array(self.grad_mean)
        self.grad_std = np.array(self.grad_std)

    def detect(self, epoch, selected, id, noisy_level_all):
        self.eval()
        top_indices_5 = np.argsort(noisy_level_all)[-5:]
        if epoch < self.args.correct_start:
            eps = self.args.eps_initial
        else:
            eps = self.args.eps_correct
        if self.args.featurelearning:
            (train_features, train_labels, train_index) = self.featureextractor(self.iter_data)
            if self.args.ext == 'classifier' and id in top_indices_5 and (self.args.correction == 1) and (epoch > self.args.correct_start) and (epoch < self.args.correct_end) and (epoch % self.args.correct_interval == 0):
                (train_acc, self_means, self_covariances, self_weights, self_subspace) = cl.evaluate('train', self.args.classifier, train_features.to(torch.float64), train_labels.to(torch.float64), num_classes=self.prop_class_inner, return_pred=False, epsilon=eps)
                for c in range(self.args.classnum):
                    if self.label_presence[c] == 0 or selected == 0:
                        self_covariances[c] = copy.deepcopy(self.covariances[c])
                    else:
                        self_covariances[c] = (self.covariances[c] - self.prop_class[c] * self_covariances[c]) / (1 - self.prop_class[c])
            else:
                self_covariances = copy.deepcopy(self.covariances)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            (test_acc, train_scores_prob, train_predmax, train_pred, train_predoriginal) = cl.evaluate('detect', self.args.classifier, train_features.to(torch.float64), train_labels.to(torch.float64), self.means, self_covariances, self.weights, self.subspace, label_presence=self.label_presence, epsilon=eps)
            del train_features, self_covariances
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            train_index = []
            with torch.no_grad():
                accuracy = 0
                for (x_batchs, y_batchs, indices) in self.iter_data:
                    train_scores_prob = self.__call__(x_batchs.to(DEVICETYPE))
                    train_scores_prob = torch.softmax(train_scores_prob, dim=1)
                    (train_predmax, train_pred) = torch.max(train_scores_prob, dim=1)
                    indices = torch.arange(train_scores_prob.size(0), device=train_scores_prob.device)
                    labels = y_batchs.to(device=train_scores_prob.device, dtype=torch.long)
                    train_predoriginal = train_scores_prob[indices, labels]
                    out = [i == 0 for i in train_pred - y_batchs.to(DEVICETYPE)]
                    accuracy += float(sum(out))
                    train_index.append(indices)
                train_index = torch.cat(train_index)
                test_acc = accuracy / len(self.iter_data) * 100
        print('Detect probability: {}, estimated label: {}--noisy label: {}--true label: {})'.format([round(x.item(), 3) for x in train_scores_prob[0, :].cpu().numpy()], train_pred[0].item(), train_labels[0].item(), self.data.dataset.train_labels[train_index[0]]))
        local_class = np.where(self.label_presence == 1)[0]
        noisy_level = 1 - test_acc / 100
        correct_list = []
        top_indices = np.argsort(noisy_level_all)[-self.args.correction_top:]
        if id in top_indices and self.args.correction == 1 and (epoch > self.args.correct_start) and (epoch < self.args.correct_end) and (epoch % self.args.correct_interval == 0):
            correctratio_i = self.args.correctratio_initial + float(epoch - self.args.correct_start) / (self.args.correct_end - self.args.correct_start) * (self.args.correctratio_final - self.args.correctratio_initial)
            correct_thres_i = self.args.correct_thres_initial + float(epoch - self.args.correct_start) / (self.args.correct_end - self.args.correct_start) * (self.args.correct_thres_final - self.args.correct_thres_initial)
            score_diff_ratio = train_predmax / train_predoriginal
            correct_list = np.where((score_diff_ratio.cpu().numpy() > correctratio_i) & (train_predmax.cpu().numpy() > correct_thres_i) & np.isin(train_pred.cpu().numpy(), local_class))[0].tolist()
            top = np.argsort(train_predmax.cpu().numpy()[correct_list])[-int(len(correct_list) * 0.5):].tolist()
            correct_list = [correct_list[i] for i in top]
            if self.args.softPi:
                corrected_new = np.array(self.data.dataset.train_soft_labels)
                corrected_new[correct_list, :] = train_scores_prob[correct_list, :]
                self.data.dataset.train_soft_labels = corrected_new
                self.data.dataset.train_noisy_labels = np.argmax(self.data.dataset.train_soft_labels, axis=1)
            else:
                corrected_new = np.array(self.data.dataset.train_noisy_labels)[train_index]
                if self.args.softlabel == 1:
                    corrected_new[correct_list] = opr_base.softcorrect(train_scores_prob, correct_list)
                else:
                    corrected_new[correct_list] = train_pred[correct_list].tolist()
                self.data.dataset.train_noisy_labels[train_index] = corrected_new
        return (noisy_level, len(correct_list))

class GlobalNetwork(BasicNetwork):

    def __init__(self, args, data, traindata=None):
        super(GlobalNetwork, self).__init__(args, data)
        self.grad_update = 0
        self.grad_basic = torch.zeros([1, 1])
        self.grad_error = torch.zeros([1, 1])
        self.grad_sparsity = torch.zeros([1, 1])
        self.grad_compress = torch.zeros([1, 1])
        self.alpha = 0
        self.traindata = traindata
        self.traindataloader = None

    def RefreshLearningRate(self, subsystemNum, epoch):
        self.lr = self.args.lr[subsystemNum]
        start = 0
        para = self.state_dict()
        for (key, value) in para.items():
            if re.search('num_batches_tracked', key) != None:
                para[key] += self.grad_update[start:start + value.numel()].view(value.size()).long()
            else:
                para[key] += self.grad_update[start:start + value.numel()].view(value.size())
            if torch.any(torch.isnan(para[key])):
                for (key_temp, value) in para.items():
                    para[key_temp] = torch.zeros_like(para[key_temp])
                break
            start += value.numel()
        self.load_state_dict(para, strict=True)
