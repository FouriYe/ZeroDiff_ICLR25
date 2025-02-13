# author: ZihanYe
# ZeroDiff (ICLR25)
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import datasets.image_util as util
from sklearn.preprocessing import MinMaxScaler
import sys
import copy
import pdb


class CLASSIFIER:
    # train_Y is interger
    def __init__(self, _train_X, _train_Y, data_loader, _nclass, _cuda, _lr=0.001, _beta1=0.5, _nepoch=20,
                 _batch_size=100, cls_mode="GZSL", useV=True, useS=False, useC=False, netDec=None, dec_size=4096,
                 dec_hidden_size=4096, _train_C=None, con_size=2048):
        self.train_X = _train_X.clone()
        self.train_Y = _train_Y.clone()
        if _train_C is not None:
            self.train_C = _train_C.clone()
        else:
            self.train_C = None
        self.test_seen_feature = data_loader.test_seen_feature.clone()
        self.test_seen_con = data_loader.test_seen_paco.clone()
        self.test_seen_label = data_loader.test_seen_label
        self.test_unseen_feature = data_loader.test_unseen_feature.clone()
        self.test_unseen_con = data_loader.test_unseen_paco.clone()
        self.test_unseen_label = data_loader.test_unseen_label
        self.seenclasses = data_loader.seenclasses
        self.unseenclasses = data_loader.unseenclasses
        self.batch_size = _batch_size
        self.nepoch = _nepoch
        self.nclass = _nclass
        self.cuda = _cuda
        self.netDec = netDec

        self.useV = useV
        self.useS = useS
        self.useC = useC

        self.input_dim = 0
        if self.useV:
            self.input_dim += _train_X.size(1)
        if self.useS:
            self.netDec.eval()
            self.input_dim += dec_size
            self.input_dim += dec_hidden_size
        if self.useC:
            self.input_dim += con_size
        self.train_X = self.compute_dec_out(self.train_X, self.input_dim, test_C=self.train_C)
        self.test_unseen_feature = self.compute_dec_out(self.test_unseen_feature, self.input_dim, test_C=self.test_unseen_con)
        self.test_seen_feature = self.compute_dec_out(self.test_seen_feature, self.input_dim, test_C=self.test_seen_con)

        self.model = LINEAR_LOGSOFTMAX_CLASSIFIER(self.input_dim, self.nclass)
        self.model.apply(util.weights_init)

        self.criterion = nn.NLLLoss()
        self.input = torch.FloatTensor(_batch_size, self.input_dim)
        self.label = torch.LongTensor(_batch_size)
        self.lr = _lr
        self.beta1 = _beta1
        self.optimizer = optim.Adam(self.model.parameters(), lr=_lr, betas=(_beta1, 0.999))
        if self.cuda:
            self.model.cuda()
            self.criterion.cuda()
            self.input = self.input.cuda()
            self.label = self.label.cuda()
        self.index_in_epoch = 0
        self.epochs_completed = 0
        self.ntrain = self.train_X.size()[0]
        if cls_mode == "GZSL":
            self.acc_seen, self.acc_unseen, self.H, self.epoch, self.best_acc_U_list, self.best_acc_S_list = self.fit()
        elif cls_mode == "ZSL":
            self.acc, self.best_model, self.best_acc_zsl_list = self.fit_zsl()
        else:
            self.acc, self.best_model, self.best_acc_seen_list = self.fit_seen()

    def fit_seen(self):
        best_acc = 0
        best_model = copy.deepcopy(self.model.state_dict())
        for epoch in range(self.nepoch):
            for i in range(0, self.ntrain, self.batch_size):
                self.model.zero_grad()
                batch_input, batch_label = self.next_batch(self.batch_size)
                self.input.copy_(batch_input)
                self.label.copy_(batch_label)

                inputv = Variable(self.input)
                labelv = Variable(self.label)
                output = self.model(inputv)
                loss = self.criterion(output, labelv)
                loss.backward()
                self.optimizer.step()
            acc_list, acc = self.val(self.test_seen_feature, self.test_seen_label, self.seenclasses)
            if acc > best_acc:
                best_acc = acc
                best_model = copy.deepcopy(self.model.state_dict())
        return best_acc, best_model, acc_list

    def fit_zsl(self):
        best_acc = -1
        best_model = copy.deepcopy(self.model.state_dict())
        for epoch in range(self.nepoch):
            for i in range(0, self.ntrain, self.batch_size):
                self.model.zero_grad()
                batch_input, batch_label = self.next_batch(self.batch_size)
                self.input.copy_(batch_input)
                self.label.copy_(batch_label)

                inputv = Variable(self.input)
                labelv = Variable(self.label)
                output = self.model(inputv)
                loss = self.criterion(output, labelv)
                loss.backward()
                self.optimizer.step()
            acc_list, acc = self.val(self.test_unseen_feature, self.test_unseen_label, self.unseenclasses)
            # print("T1: ", acc)
            if acc > best_acc:
                best_acc = acc
                best_model = copy.deepcopy(self.model.state_dict())
                best_acc_list = acc_list
        return best_acc, best_model, best_acc_list

    def fit(self):
        best_H = -1
        best_seen = -1
        best_unseen = -1
        out = []
        best_model = copy.deepcopy(self.model.state_dict())
        # early_stopping = EarlyStopping(patience=20, verbose=True)
        for epoch in range(self.nepoch):
            for i in range(0, self.ntrain, self.batch_size):
                self.model.zero_grad()
                batch_input, batch_label = self.next_batch(self.batch_size)
                self.input.copy_(batch_input)
                self.label.copy_(batch_label)

                inputv = Variable(self.input)
                labelv = Variable(self.label)
                output = self.model(inputv)
                loss = self.criterion(output, labelv)
                loss.backward()
                self.optimizer.step()
            acc_seen = 0
            acc_unseen = 0
            acc_seen_list, acc_seen = self.val_gzsl(self.test_seen_feature, self.test_seen_label, self.seenclasses)
            acc_unseen_list, acc_unseen = self.val_gzsl(self.test_unseen_feature, self.test_unseen_label,
                                                        self.unseenclasses)
            H = 2 * acc_seen * acc_unseen / (acc_seen + acc_unseen)
            # print("H: ", H)
            if H > best_H:
                best_seen = acc_seen
                best_unseen = acc_unseen
                best_H = H
                best_acc_seen_list = acc_seen_list
                best_acc_unseen_list = acc_unseen_list
        return best_seen, best_unseen, best_H, epoch, best_acc_unseen_list, best_acc_seen_list

    def next_batch(self, batch_size):
        start = self.index_in_epoch
        # shuffle the data at the first epoch
        if self.epochs_completed == 0 and start == 0:
            perm = torch.randperm(self.ntrain)
            self.train_X = self.train_X[perm]
            self.train_Y = self.train_Y[perm]
        # the last batch
        if start + batch_size > self.ntrain:
            self.epochs_completed += 1
            rest_num_examples = self.ntrain - start
            if rest_num_examples > 0:
                X_rest_part = self.train_X[start:self.ntrain]
                Y_rest_part = self.train_Y[start:self.ntrain]
            # shuffle the data
            perm = torch.randperm(self.ntrain)
            self.train_X = self.train_X[perm]
            self.train_Y = self.train_Y[perm]
            # start next epoch
            start = 0
            self.index_in_epoch = batch_size - rest_num_examples
            end = self.index_in_epoch
            X_new_part = self.train_X[start:end]
            Y_new_part = self.train_Y[start:end]
            # print(start, end)
            if rest_num_examples > 0:
                return torch.cat((X_rest_part, X_new_part), 0), torch.cat((Y_rest_part, Y_new_part), 0)
            else:
                return X_new_part, Y_new_part
        else:
            self.index_in_epoch += batch_size
            end = self.index_in_epoch
            # print(start, end)
            # from index start to index end-1
            return self.train_X[start:end], self.train_Y[start:end]

    def val_gzsl(self, test_X, test_label, target_classes):
        start = 0
        ntest = test_X.size()[0]
        predicted_label = torch.LongTensor(test_label.size())
        for i in range(0, ntest, self.batch_size):
            end = min(ntest, start + self.batch_size)
            with torch.no_grad():
                if self.cuda:
                    inputX = Variable(test_X[start:end].cuda())
                else:
                    inputX = Variable(test_X[start:end])
            output = self.model(inputX)
            _, predicted_label[start:end] = torch.max(output.data, 1)
            start = end

        acc_list, acc = self.compute_per_class_acc_gzsl(test_label, predicted_label, target_classes)
        return acc_list, acc

    def compute_per_class_acc_gzsl(self, test_label, predicted_label, target_classes):
        nclass = target_classes.shape[0]
        acc_per_class = torch.FloatTensor(nclass).fill_(0)
        for i in range(nclass):
            c = target_classes[i]
            idx = (test_label == c)
            acc_per_class[i] = torch.sum(test_label[idx] == predicted_label[idx]) / torch.sum(idx)
        # acc_per_class /= target_classes.size(0)
        return acc_per_class, acc_per_class.mean()

        # test_label is integer

    def val(self, test_X, test_label, target_classes):
        start = 0
        ntest = test_X.size()[0]
        predicted_label = torch.LongTensor(test_label.size())
        for i in range(0, ntest, self.batch_size):
            end = min(ntest, start + self.batch_size)
            with torch.no_grad():
                if self.cuda:
                    inputX = Variable(test_X[start:end].cuda())
                else:
                    inputX = Variable(test_X[start:end])
            output = self.model(inputX)
            _, predicted_label[start:end] = torch.max(output.data, 1)
            start = end

        acc_list, acc = self.compute_per_class_acc(util.map_label(test_label, target_classes), predicted_label,
                                                   target_classes.size(0))
        return acc_list, acc

    def compute_per_class_acc(self, test_label, predicted_label, nclass):
        acc_per_class = torch.FloatTensor(nclass).fill_(0)
        for i in range(nclass):
            idx = (test_label == i)
            acc_per_class[i] = torch.sum(test_label[idx] == predicted_label[idx]) / torch.sum(idx)
        return acc_per_class, acc_per_class.mean()

    def compute_dec_out(self, test_X, new_size, test_C=None):
        start = 0
        ntest = test_X.size()[0]
        new_test_X = torch.zeros(ntest, new_size)
        for i in range(0, ntest, self.batch_size):
            end = min(ntest, start + self.batch_size)
            with torch.no_grad():
                if self.cuda:
                    inputX = Variable(test_X[start:end].cuda())
                else:
                    inputX = Variable(test_X[start:end])
            inputData = []
            if self.useV:
                inputData.append(inputX)
            if self.useS:
                feat1 = self.netDec(inputX)
                feat2 = self.netDec.getLayersOutDet()
                inputData.append(feat1)
                inputData.append(feat2)
            if self.useC:
                if self.cuda:
                    inputC = Variable(test_C[start:end].cuda())
                else:
                    inputC = Variable(test_C[start:end])
                inputData.append(inputC)
            new_test_X[start:end] = torch.cat(inputData, dim=1).data.cpu()
            start = end
        return new_test_X


class LINEAR_LOGSOFTMAX_CLASSIFIER(nn.Module):
    def __init__(self, input_dim, nclass):
        super(LINEAR_LOGSOFTMAX_CLASSIFIER, self).__init__()
        self.fc = nn.Linear(input_dim, nclass)
        self.logic = nn.LogSoftmax(dim=1)

    def forward(self, x):
        o = self.logic(self.fc(x))
        return o
