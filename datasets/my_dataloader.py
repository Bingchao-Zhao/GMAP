
import random
import torch
import pandas as pd
from tqdm import tqdm

import torch.utils.data as data
from my_utils.utils import *
import my_utils.file_util as fu
import h5py

LABEL = {'1p19q':['codel', 'non-codel'],
        '7g10l':['Gain chr 7 & loss chr 10', 'No combined CNA'],
        'TERT':['Mutant', 'WT'],
        'IDH':['Mutant', 'WT']}

DATA_DIR = {
    'TCGA' : {'UNI': "feature_path",
              "ResNet50":"feature_path"},
    
}

TOTAL_LABEL_PATH = 'label/total_label.csv'

def select_dataloader(dataset_name='TCGA', mod='train', gen_type='ATRX',extroctor='UNI'):
    if dataset_name=='TCGA':
        return TCGA_datset(mod=mod, gen_type=gen_type,
                           dataset=dataset_name,extroctor=extroctor)


def read_total_label(label_path, dataset,gen):
    gen_dict = {'TERT':"TERT", 
                'IDH':"IDH", 
                '1p19q':"1p/19q", 
                '7g10l':"chr7_gain_chr10_loss"}
    data = pd.read_csv(label_path)
    id_list = data['Path_id'][data['cohort']==dataset].tolist()
    gen_event = data[gen_dict[gen]][data['cohort']==dataset].tolist()
    return  {id_list[i].lower():gen_event[i] for i in range(len(id_list))}


def get_label(value, label):
    # pos, neg = LABEL[gen_type][0], LABEL[gen_type][1]
    
    if label.get(value.lower()) is None: return None
    if np.isnan(float(label.get(value.lower()))):
        return None
    elif int(label[value.lower()])==1: return 1
    elif int(label[value.lower()])==0: return 0
    else: 
        err(f"error")

def load_h5(data_path):
    file = h5py.File(data_path, 'r')
 
    features = file['features']
    features = features[()]
    coords = file['coords']
    coords = coords[()]
    return features, coords


def load_data(gen_type,
              load_data_list, 
              labe_record, 
              load_patient=None, 
              dataset='TCGA', 
              mod='train'):
    pos_num, neg_num, nolabel_num = 0, 0, 0
    data,load_patient = [],[]
    flow('*'*20, f"Loading '{dataset}' {gen_type} '{mod}' data!! File num:{len(load_data_list)}.", '*'*20)
    for [h5, patient_name] in tqdm(load_data_list):
        label = labe_record[patient_name.lower()]
        if label==1: pos_num+=1
        elif label==0: neg_num+=1
        else: 
            nolabel_num +=1 
            continue

        if patient_name not in load_patient:
            load_patient.append(patient_name)
            
        features, coords = load_h5(h5)
        data.append([torch.stack([torch.FloatTensor(i) for i in features]), 
                            torch.stack([torch.LongTensor(i) for i in coords]),
                            patient_name, h5])
    flow(f"'{dataset}' {gen_type} '{mod}'.",
         f"Total patient num:{len(load_patient)}." ,
         f"Total WSI num:{len(data)}. Pos num:{pos_num}, neg num: {neg_num}, nolabel_num:{nolabel_num}")
    if len(data)==0:
            err(f"'{dataset}' dataset has no '{gen_type}' label!")
    return data        

class TCGA_datset(data.Dataset):
    def __init__(self,mod='train', gen_type='ATRX', dataset='TCGA',extroctor="UNI"):
        flow('$'*30,f"dataset:{dataset}, gen_type:{gen_type},  FEATURE_EXTRACTOR:{extroctor}",'$'*30)
        self.gen_type = gen_type
        # self.label = read_label(self.label_path)
        self.label = read_total_label(TOTAL_LABEL_PATH, dataset, gen_type)
        self.mod = mod
        self.data = []
        self.load_patient = []
        
        self.total_data = find_file(DATA_DIR[dataset][extroctor], 1, suffix='.h5')#TCGA-02-0001
        self.svs_to_patient = {get_name_from_path(i):get_name_from_path(i)[0:12].lower() for i in self.total_data}
        self.pos_num = 0
        self.neg_num = 0

        if mod == "train":
            self.patient_name_list = fu.csv_reader('data_split/train.csv')
        elif mod == "val":
            self.patient_name_list = fu.csv_reader('data_split/val.csv')
        elif mod == "test":
            self.patient_name_list = fu.csv_reader('data_split/test.csv')
        self.patient_name_list = [i[0] for i in self.patient_name_list]
        random.shuffle(self.patient_name_list)
        self.load_data_list = []

        no_label_data = []
        for h5 in self.total_data:
            wsi_name = get_name_from_path(h5)
            patient_name = '-'.join(wsi_name.split('-')[0:3])
            if get_label(patient_name, self.label) is None: 
                no_label_data.append(patient_name)
                continue
            if patient_name in self.patient_name_list:
                self.load_data_list.append([h5, patient_name])
            if len(self.load_data_list)>=5: break
        flow(f"'{dataset}' {gen_type} '{mod}' total svs num:{len(self.total_data)}. no label data num: {len(no_label_data)}")
            
        self.data = load_data(gen_type,
                                self.load_data_list, 
                                self.label, 
                                load_patient=None, 
                                dataset=dataset, 
                                mod=mod)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        features, patch_index, patient_name,h5_path  = self.data[idx]

        label = get_label(patient_name.lower(), self.label)
        if self.mod == 'train':
            index = [x for x in range(features.shape[0])]

            random.shuffle(index)
            features = features[index]
            patch_index = patch_index[index]

        return features, label, patch_index, h5_path, get_name_from_path(h5_path) 
    



