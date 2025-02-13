#import h5py
import numpy as np
import scipy.io as sio
import torch
from sklearn import preprocessing
import torch.nn.functional as F

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def map_label(label, classes):
    mapped_label = torch.LongTensor(label.size())
    for i in range(classes.size(0)):
        mapped_label[label==classes[i]] = i    

    return mapped_label

def preprocessing_fun(opt, train_feature, test_seen_feature, test_unseen_feature):
    if not opt.validation:
        if opt.preprocessing:
            if opt.standardization:
                print('standardization...')
                scaler = preprocessing.StandardScaler()
            else:
                scaler = preprocessing.MinMaxScaler()

            train_feature = scaler.fit_transform(train_feature)
            test_seen_feature = scaler.transform(test_seen_feature)
            test_unseen_feature = scaler.transform(test_unseen_feature)
            train_feature = torch.from_numpy(train_feature).float()
            mx = train_feature.max()
            train_feature.mul_(1 / mx)

            test_unseen_feature = torch.from_numpy(test_unseen_feature).float()
            test_unseen_feature.mul_(1 / mx)
            test_seen_feature = torch.from_numpy(test_seen_feature).float()
            test_seen_feature.mul_(1 / mx)
        else:
            train_feature = torch.from_numpy(train_feature).float()
            test_unseen_feature = torch.from_numpy(test_unseen_feature).float()
            test_seen_feature = torch.from_numpy(test_seen_feature).float()
    else:
        train_feature = torch.from_numpy(train_feature).float()
        test_unseen_feature = torch.from_numpy(test_unseen_feature).float()

    return train_feature, test_seen_feature, test_unseen_feature


class DATA_LOADER(object):
    def __init__(self, opt):
        self.read_matdataset(opt)
        self.index_in_epoch = 0
        self.epochs_completed = 0

    def read_matdataset(self, opt):
        matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/res101.mat")
        feature = matcontent['features'].T
        label = matcontent['labels'].astype(int).squeeze() - 1
        matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.class_embedding + "_splits.mat")
        self.allclasses_names = matcontent['allclasses_names'].squeeze()
        trainval_loc = matcontent['trainval_loc'].squeeze() - 1
        train_loc = matcontent['train_loc'].squeeze() - 1
        val_unseen_loc = matcontent['val_loc'].squeeze() - 1
        test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
        test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1

        if opt.class_embedding == "att":
            if opt.dataset == "FLO":
                self.ori_attribute = torch.from_numpy(matcontent['att'].T).float()
            else:
                self.ori_attribute = torch.from_numpy(matcontent['original_att'].T).float()
            self.attribute = torch.from_numpy(matcontent['att'].T).float()
        elif opt.class_embedding == "sent":
            self.ori_attribute = torch.from_numpy(matcontent['att'].T).float()
            self.attribute = torch.from_numpy(matcontent['att'].T).float()

        if opt.class_embedding_norm:
            self.attribute = F.normalize(self.ori_attribute, p=2, dim=1)
        else:
            self.attribute = self.ori_attribute

        matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/ce_ce.mat")
        feature = matcontent['features']

        self.train_label = torch.from_numpy(label[trainval_loc]).long()
        self.test_unseen_label = torch.from_numpy(label[test_unseen_loc]).long()
        self.test_seen_label = torch.from_numpy(label[test_seen_loc]).long()
        self.train_feature = torch.from_numpy(feature[trainval_loc]).float()
        self.test_unseen_feature = torch.from_numpy(feature[test_unseen_loc]).float()
        self.test_seen_feature = torch.from_numpy(feature[test_seen_loc]).float()
        self.train_feature, self.test_seen_feature, self.test_unseen_feature = preprocessing_fun(opt, self.train_feature, self.test_seen_feature, self.test_unseen_feature)

        matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/con_paco.mat")
        paco_feature = matcontent['features']
        self.train_paco = torch.from_numpy(paco_feature[trainval_loc]).float()
        self.test_unseen_paco = torch.from_numpy(paco_feature[test_unseen_loc]).float()
        self.test_seen_paco = torch.from_numpy(paco_feature[test_seen_loc]).float()
        self.train_paco, self.test_seen_paco, self.test_unseen_paco = preprocessing_fun(opt, self.train_paco, self.test_seen_paco, self.test_unseen_paco)

        if opt.split_percent != 100:
            split_matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/split_"+str(opt.split_percent)+"percent.mat")
            trainval_loc_split = split_matcontent['trainval_loc'].squeeze() - 1
            self.train_feature = self.train_feature[trainval_loc_split]
            self.train_paco = self.train_paco[trainval_loc_split]
            self.train_label = self.train_label[trainval_loc_split]
    
        self.seenclasses = torch.from_numpy(np.unique(self.train_label.numpy()))
        self.unseenclasses = torch.from_numpy(np.unique(self.test_unseen_label.numpy()))

        self.ntrain = self.train_feature.size()[0]
        self.ntest_seen = self.test_seen_feature.size()[0]
        self.ntest_unseen = self.test_unseen_feature.size()[0]
        self.nclass_seen = self.seenclasses.size(0)
        self.nclass_unseen = self.unseenclasses.size(0)
        self.train_class = self.seenclasses.clone()
        self.allclasses = torch.arange(0, self.nclass_seen + self.nclass_unseen).long()
        self.train_mapped_label = map_label(self.train_label, self.seenclasses)
        self.attribute_seen = self.attribute[self.seenclasses]

    def next_seen_batch(self, batch_size):

        idx = torch.randperm(self.ntrain)[0:batch_size]
        batch_feature = self.train_feature[idx]
        batch_con = self.train_paco[idx]
        batch_label = self.train_label[idx]
        batch_att = self.attribute[batch_label]

        return batch_feature, batch_con, batch_att, batch_label


    def next_test_seen_batch(self, seen_batch):
        idx = torch.randperm(self.ntest_seen)[0:seen_batch]
        batch_feature = self.test_seen_feature[idx]
        batch_con = self.test_seen_paco[idx]
        batch_label = self.test_seen_label[idx]
        batch_att = self.attribute[batch_label]
        return batch_feature, batch_con, batch_att, batch_label

