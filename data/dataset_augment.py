import os
import faiss
import bdpy
import json
import glob
import random
import numpy as np
import pandas as pd
from PIL import Image
import torch.utils.data
import data.configure as config
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.io.image as imageio

def DIR_sub_without_images(sub = 'sub-3', rois = ['ROI_VC']):
    train_rois, test_rois = [], []
    DIR_sub_train = bdpy.BData(os.path.join(config.DIR_dataset_dir, config.DIR_train_subs[sub]))
    DIR_sub_test = bdpy.BData(os.path.join(config.DIR_dataset_dir, config.DIR_test_subs[sub]))

    train_image_index = DIR_sub_train.select('image_index').squeeze().astype(int) - 1
    test_image_index = DIR_sub_test.select('image_index').squeeze().astype(int) - 1

    trainStiIDs = np.array(pd.read_csv(config.kamitani_sti_trainID, header = None)[1])[train_image_index]
    testStiIDs = np.array(pd.read_csv(config.kamitani_sti_testID, header = None)[1])
    
    MAX_DIM = 0
    for roi in rois:
        train_roi_fMRI = DIR_sub_train.select(roi)
        test_roi_fMRI = DIR_sub_test.select(roi)

        test_roi_fMRI_avg = np.zeros([50, test_roi_fMRI.shape[1]])
        for i in range(50):
            test_roi_fMRI_avg[i] = np.mean(test_roi_fMRI[test_image_index == i], axis = 0)

        train_rois.append(train_roi_fMRI)
        test_rois.append(test_roi_fMRI_avg)
        MAX_DIM = train_roi_fMRI.shape[-1] if train_roi_fMRI.shape[-1] > MAX_DIM else MAX_DIM

    train_rois = np.concatenate(([np.pad(fmri, ((0, 0), (0, MAX_DIM - fmri.shape[-1])))[:,None,:] for fmri in train_rois]), 1).squeeze()
    test_rois = np.concatenate(([np.pad(fmri, ((0, 0), (0, MAX_DIM - fmri.shape[-1])))[:,None,:] for fmri in test_rois]), 1).squeeze()

    train_cat_rois = {}
    trainCatIDs = [id.split('_')[0] for id in trainStiIDs]
    trainCatSet = set(trainCatIDs)
    for cat in trainCatSet:
        train_cat_rois[cat] = train_rois[np.array(trainCatIDs) == cat]

    test_cat_rois = {}
    testCatIDs = [id.split('_')[0] for id in testStiIDs]
    for cat in testCatIDs:
        test_cat_rois[cat] = test_rois[np.array(testCatIDs) == cat]

    return train_cat_rois, train_rois, trainStiIDs, test_cat_rois, test_rois, testStiIDs

class Visual_Text_fMRI_Dataset(Dataset):
     def __init__(self, imageIDs, caps, categories, fMRI, tokenizer, mixup = False, train = True, 
                  transform = None, max_caption_length = 25, clip_features = None):
        self.tokenizer = tokenizer
        self.imageIDs = imageIDs
        self.fMRI = fMRI
        self.mixup = mixup
        self.train = train
        self.transform = transform
        self.caps = caps
        self.categories = categories
        self.clip_features = clip_features
        self.SIMPLE_PREFIX = "This image shows "
        self.retrieved_caps = None
        self.CAPTION_LENGTH = max_caption_length

        self.template = self.SIMPLE_PREFIX
        self.max_target_length = (max_caption_length
                                + len(tokenizer.encode(self.template)))

     def __len__(self):
        return len(self.imageIDs)

     def prep_strings(self, text, tokenizer, retrieved_caps = None): 
        if not self.train:
            padding = False
            truncation = False
        else:
            padding = True 
            truncation = True
        
        if retrieved_caps is not None:
            infix = '\n\n'.join(retrieved_caps) + '.'
            prefix = self.template.replace('||', infix)
        else:
            prefix = self.SIMPLE_PREFIX

        prefix_ids = tokenizer.encode(prefix)
        len_prefix = len(prefix_ids)

        text_ids = tokenizer.encode(text, add_special_tokens = False)
        if truncation:
            text_ids = text_ids[:self.CAPTION_LENGTH]
        input_ids = prefix_ids + text_ids if self.train else prefix_ids

        # we ignore the prefix (minus one as the first subtoken in the prefix is not predicted)
        label_ids = [-100] * (len_prefix - 1) + text_ids + [tokenizer.eos_token_id] 
        if padding:
            input_ids += [tokenizer.pad_token_id] * (self.max_target_length - len(input_ids))
            label_ids += [-100] * (self.max_target_length - len(label_ids))
        
        if not self.train:
            return input_ids
        else:  
            return input_ids, label_ids
     
     
     def __getitem__(self, idx):
        file_name = self.imageIDs[idx].split('/')[-1]
        category_id = file_name.split('_')[0]
        image_id = file_name.split('_')[1].split('.')[0]
        #image = Image.open(self.imageIDs[idx])
        cap = self.caps[file_name]
        #category = self.categories[category_id]
        #visual_features = self.clip_features[idx]
        visual_features = self.clip_features[file_name][()]
        if(self.train):
            fmri_num = self.fMRI[category_id].shape[0]
            selected_no = np.random.permutation(range(fmri_num))[:random.randint(1, fmri_num - 1)]
            if(self.mixup and selected_no.shape[0] != 1):
                coefficient = torch.tensor(np.random.uniform(-1, 1, size = selected_no.shape[0])).softmax(0)
                selected_fMRI = torch.tensor(self.fMRI[category_id][selected_no])
                coefficient = coefficient[:, None, None] if len(selected_fMRI.shape) == 3 else coefficient[:, None]
                mixup_fMRI = torch.sum(selected_fMRI * coefficient, 0)
                fMRI = mixup_fMRI.numpy()
            else:
                fMRI = self.fMRI[category_id][selected_no[0]].squeeze()
        else:
            selected = random.randint(0, self.fMRI[category_id].shape[0] - 1)
            fMRI = self.fMRI[category_id][selected]

        #image = self.transform(image) if self.transform else image
        k_caption = None
        decoder_input_ids, labels = self.prep_strings(cap, self.tokenizer, k_caption)
        data = {'encoder_inputs': fMRI.astype(np.float32), 'encoder_labels': visual_features, 
                'decoder_input_ids': np.array(decoder_input_ids), 'decoder_labels': np.array(labels)}
        return data

def get_visual_text_dir_dataset(sub, rois, batch_size, mixup = True, candidate = True, 
                            tokenizer = None, clip_features = None):

    train_cat_rois, _, trainStiIDs,\
    test_cat_rois, _, testStiIDs = DIR_sub_without_images(sub, rois)
    
    fmri_dim = train_cat_rois[trainStiIDs[0].split('_')[0]].shape[-1]
    Train_category = set([id.split('_')[0] for id in trainStiIDs])
    if(candidate):
        train_images = np.concatenate([np.array(glob.glob(f"{config.kamitani_Tr_Aug}/{category}/*.JPEG")) for category in Train_category])
    else:
        train_images = np.concatenate([np.array(glob.glob(f"{config.kamitani_Tr_Aug}/*/{image}")) for image in trainStiIDs])
        
    train_caps = json.load(open(config.smallCap_Kamitani_train))
    #classIDs = np.array(pd.read_csv(config.kamitani_sti_text, header = None)[0])
    #classTexts = np.array(pd.read_csv(config.kamitani_sti_text, header = None)[1])
    #id_text = dict(zip(classIDs, classTexts))

    train_dataset = Visual_Text_fMRI_Dataset(train_images, train_caps, None, train_cat_rois, tokenizer, mixup, 
                                                True, clip_features = clip_features)
    train_dataloader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
    return train_dataset, train_dataloader, fmri_dim