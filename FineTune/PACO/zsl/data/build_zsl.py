from os.path import join

import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
import numpy as np
from sklearn import preprocessing

from .test_dataset import TestDataset

from .transforms.data_transform import data_transform_fun

import scipy.io as sio
import copy

from einops import rearrange, reduce, repeat
from sklearn.cluster import KMeans

from PIL import Image

class RandDataset(data.Dataset):

    def __init__(self, img_path, atts, labels, transforms=None):
        self.img_path = img_path
        self.atts = torch.tensor(atts).float()
        self.labels = torch.tensor(labels).long()
        self.classes = np.unique(labels)

        self.transforms = transforms

    def __getitem__(self, index):
        img_path = self.img_path[index]
        img = Image.open(img_path).convert('RGB')

        if self.transforms is not None:
            img = self.transforms(img) # "resize_random_crop"

        label = self.labels[index]
        att = self.atts[index]

        return img, att, label

    def __len__(self):
        return self.labels.size(0)

class ImgDatasetParam(object):
    DATASETS = {
        "imgroot": '/Data_PHD_Backup/phd22_zihan_ye/datasets',
        "dataroot": '/Data_PHD_Backup/phd22_zihan_ye/datasets/xlsa17/data',
        "image_embedding": 'res101',
        "class_embedding": 'att'
    }

    @staticmethod
    def get(dataset):
        attrs = ImgDatasetParam.DATASETS
        attrs["imgroot"] = join(attrs["imgroot"], dataset)
        args = dict(
            dataset=dataset
        )
        args.update(attrs)
        return args

def read_att_name(dataset, dataset_path):
    att_names = []

    if dataset =="AWA2":
        with open(dataset_path+"/predicates.txt",'r') as read_file_path:
            while True:
                lines = read_file_path.readline()
                if not lines:
                    break
                att_name = lines.replace('\t','').replace('\n','').strip()
                print(att_name)
                att_names.append(att_name)
    return att_names


def split_train_val(train_data, ratio=0.8):
    train_img, train_att, train_label = (train_data)
    num_train = train_img.shape[0]
    permutated_id = np.random.permutation(np.arange(num_train))
    selected_tr_id = permutated_id[:int(num_train*ratio)]
    selected_val_id = permutated_id[int(num_train*ratio):]
    train_data = (train_img[selected_tr_id], train_att[selected_tr_id], train_label[selected_tr_id])
    val_data = (train_img[selected_val_id], train_att[selected_val_id], train_label[selected_val_id])
    

    return train_data, val_data

def build_dataloader(cfg):

    args = ImgDatasetParam.get(cfg.DATASETS_NAME)
    imgroot = args['imgroot']
    dataroot = args['dataroot']
    image_embedding = args['image_embedding']
    class_embedding = args['class_embedding']
    dataset = args['dataset']

    matcontent = sio.loadmat(dataroot + "/" + dataset + "/" + image_embedding + ".mat")

    img_files =np.squeeze(matcontent['image_files'])
    new_img_files = []
    for img_file in img_files:
        img_path = img_file[0]
        if dataset=='CUB':
            img_path = imgroot[:-4] + img_path.split("MSc/CUB_200_2011")[1]
        elif dataset=='AWA2':
            img_path = imgroot[:-4] + '/Animals_with_Attributes2/JPEGImages' + img_path.split("JPEGImages")[1]
        elif dataset=='SUN':
            img_path = join(imgroot, img_path.split("SUN/")[1])
        new_img_files.append(img_path)

    new_img_files = np.array(new_img_files)
    label = matcontent['labels'].astype(int).squeeze() - 1

    matcontent = sio.loadmat(dataroot + "/" + dataset + "/" + class_embedding + "_splits.mat")
    trainvalloc = matcontent['trainval_loc'].squeeze() - 1
    test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
    test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1
    cls_name = matcontent['allclasses_names']
    if cfg.DATASETS_SEMANTIC_TYPE =="GBU":
        if cfg.DATASETS_SEMANTIC == "normalized":
            att_name = 'att'
        elif cfg.DATASETS.SEMANTIC == "original":
            att_name = 'original_att'
        else:
            print('unrecognized SEMANTIC')
            att_name = 'att'
        attribute = matcontent[att_name].T
    else:
        print('unrecognized SEMANTIC TYPE')

    train_img = new_img_files[trainvalloc]
    train_label = label[trainvalloc].astype(int)
    train_att = attribute[train_label]

    train_id_unmapped, idx = np.unique(train_label, return_inverse=True)
    train_att_unique = attribute[train_id_unmapped]
    train_clsname = cls_name[train_id_unmapped]

    num_train = len(train_id_unmapped)
    train_label = idx
    train_id = np.unique(train_label)

    test_img_unseen = new_img_files[test_unseen_loc]
    test_label_unseen = label[test_unseen_loc].astype(int)
    test_id_unmapped, idx = np.unique(test_label_unseen, return_inverse=True)
    att_unseen = attribute[test_id_unmapped]
    test_clsname = cls_name[test_id_unmapped]
    test_label_unseen = idx + num_train
    test_id = np.unique(test_label_unseen)

    cls_name_mapped = np.concatenate([train_clsname,test_clsname])

    train_test_att = np.concatenate((train_att_unique, att_unseen))
    train_test_id = np.concatenate((train_id, test_id))

    test_img_seen = new_img_files[test_seen_loc]
    test_label_seen = label[test_seen_loc].astype(int)
    _, idx = np.unique(test_label_seen, return_inverse=True)
    test_label_seen = idx

    seenclasses = np.unique(train_label)
    unseenclasses = np.unique(test_label_unseen)

    n_seenclass = seenclasses.shape[0]
    n_unseenclass = unseenclasses.shape[0]
    n_allclass = n_seenclass+n_unseenclass


    att_unseen = torch.from_numpy(att_unseen).float()
    test_label_seen = torch.tensor(test_label_seen)
    test_label_unseen = torch.tensor(test_label_unseen)
    train_label = torch.tensor(train_label)

    att_seen = torch.from_numpy(train_att_unique).float()
    att_all = torch.from_numpy(attribute).float()

    att_name_all = matcontent['allclasses_names']

    res = {
        'dataset_name': dataset,
        'train_label': train_label,
        'train_att': train_att,
        'test_label_seen': test_label_seen,
        'test_label_unseen': test_label_unseen,
        'att_unseen': att_unseen,
        'att_seen': att_seen,
        'att_all': att_all,
        'att_name_all': att_name_all,
        'train_id': train_id,
        'test_id': test_id,
        'train_id_unmapped': train_id_unmapped,
        'test_id_unmapped': test_id_unmapped,
        'train_test_id': train_test_id,
        'train_clsname': train_clsname,
        'test_clsname': test_clsname,
        "all_clsname": cls_name,
        "all_clsname_mapped": cls_name_mapped,
        'seenclasses': seenclasses,
        'unseenclasses': unseenclasses,
    }

    # train dataloader
    # ways = cfg.DATASETS.WAYS
    # shots = cfg.DATASETS.SHOTS
    data_aug_train =  "resize_random_crop"
    img_size = 224
    transforms = data_transform_fun(data_aug_train, size=img_size)

    tr_data, val_data  = split_train_val(train_data=(train_img, train_att, train_label), ratio=0.8)

    train_img, train_att, train_label = tr_data
    val_img, val_att, val_label = val_data

    val_data = TestDataset(val_img, val_att, val_label, transforms)
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=cfg.BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=False)

    all_img = new_img_files
    all_label = label.astype(int)
    all_att = attribute[all_label]
    all_data = TestDataset(all_img, all_att, all_label, transforms)
    extract_loader = torch.utils.data.DataLoader(
        all_data, batch_size=cfg.BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=False)

    dataset = RandDataset(train_img, train_att, train_label, transforms)

    sampler = torch.utils.data.sampler.RandomSampler(dataset)
    # batch = ways*shots
    # batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, batch_size=batch, drop_last=True)

    tr_dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        num_workers=8,
        # batch_sampler=batch_sampler,
        batch_size=cfg.BATCH_SIZE,
    )

    data_aug_test = "resize_crop"
    transforms = data_transform_fun(data_aug_test, size=img_size)
    test_batch_size = 32

    # test unseen dataloader
    tu_data = TestDataset(test_img_unseen, att_unseen, test_label_unseen, transforms, atts_offset=-1*num_train)
    tu_loader = torch.utils.data.DataLoader(
        tu_data, batch_size=test_batch_size, shuffle=False,
        num_workers=4, pin_memory=False)

    # test seen dataloader
    ts_data = TestDataset(test_img_seen, att_seen, test_label_seen, transforms)
    ts_loader = torch.utils.data.DataLoader(
        ts_data, batch_size=test_batch_size, shuffle=False,
        num_workers=4, pin_memory=False)


    return tr_dataloader, val_loader, extract_loader, tu_loader, ts_loader, res

def map_label(label, classes):
    # original label -> seen or unseen label index
    mapped_label = torch.LongTensor(len(label))
    for i in range(classes.size(0)):
        mapped_label[label==classes[i]] = i
    return mapped_label
