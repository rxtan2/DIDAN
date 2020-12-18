import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import pickle
import json
import random
import math
import bisect
from torch.utils.data import Dataset

class Loader(Dataset):
    def __init__(self, args, split, captioning_dataset, art2id, fake_articles):
        self.args = args
        self.split = split
        self.img_feats_dir = args.image_representations_dir
        self.real_arts_dir = args.real_articles_dir
        self.fake_arts_dir = args.fake_articles_dir
        self.real_caps_dir = args.real_captions_dir
        self.real_arts = torch.load(os.path.join(self.real_arts_dir, split + '.bert.pt'))
        self.fake_arts = torch.load(os.path.join(self.fake_arts_dir, split + '.bert.pt'))
        self.real_caps = torch.load(os.path.join(self.real_caps_dir, split + '.bert.pt'))
   
        self.ner_dir = args.ner_dir

        self.captioning_dataset = captioning_dataset
        self.art2id = art2id
        self.fake_articles = fake_articles

        self.realarts2id, self.caps2id, self.arts2caps, self.fakearts2id = self.parse()
        self.arts = []
        for i in self.realarts2id:
            name = '1_' + i
            self.arts.append(name)

        if split == 'train':
            neg_count = 0
            for i in self.fakearts2id:
                name = '0_' + i
                self.arts.append(name)
                neg_count += 1
            return

        for i in self.fakearts2id:
            name = '0_' + i
            self.arts.append(name)


    def parse(self):
        realarts2id = {}
        for i, d in enumerate(self.real_arts):
            name = d['name']
            realarts2id[name] = i

        caps2id = {}
        arts2caps = {}
        for i, d in enumerate(self.real_caps):
            name = d['name']
            art = name.split('_')[0]
            if art not in arts2caps:
                arts2caps[art] = []
            arts2caps[art].append(name)
            caps2id[name] = i
         
        fakearts2id = {}
        for i, d in enumerate(self.fake_arts):
            name = d['name']
            fakearts2id[name] = i

        return realarts2id, caps2id, arts2caps, fakearts2id

    def _pad(self, data, pad_id, width=-1):
        if (width == -1):
            width = max(len(d) for d in data)
        rtn_data = [d + [pad_id] * (width - len(d)) for d in data]
        return rtn_data

    def preprocess(self, ex):
        src = ex['src']
        tgt = ex['tgt'][:self.args.max_tgt_len][:-1]+[2]
        src_sent_labels = ex['src_sent_labels']
        segs = ex['segs']
        if(not self.args.use_interval):
            segs=[0]*len(segs)
        clss = ex['clss']
        src_txt = ex['src_txt']
        tgt_txt = ex['tgt_txt']

        end_id = [src[-1]]

        tmp = src[:-1][:self.args.max_pos - 1] + end_id

        src = src[:-1][:self.args.max_pos - 1] + end_id
        segs = segs[:self.args.max_pos]
        max_sent_id = bisect.bisect_left(clss, self.args.max_pos)
        src_sent_labels = src_sent_labels[:max_sent_id]
        clss = clss[:max_sent_id]

        return src, tgt, segs, clss, src_sent_labels

    def __getitem__(self, index):
        # Gets real article
        art = self.arts[index]
        label = int(art.split('_')[0])
        art_id = art.split('_')[-1]

        img_dict = self.captioning_dataset[art_id]['images']
        
        if label == 1:
            idx = self.realarts2id[art_id]
            art = self.real_arts[idx]
            art = self.preprocess(art)
            art_path = os.path.join(self.ner_dir, '1_' + art_id + '.pkl')
            art_ner = pickle.load(open(art_path, 'rb'))
        else:
            idx = self.fakearts2id[art_id]
            art = self.fake_arts[idx]
            art = self.preprocess(art)
            art_path = os.path.join(self.ner_dir, '0_' + art_id + '.pkl')
            art_ner = pickle.load(open(art_path, 'rb'))
    
        # Get images and captions
        imgs = self.arts2caps[art_id]
        if len(imgs) > 3:
            imgs = imgs[:3]
        combined_feats = []
        combined_caps = []
        cap_text = []
        for i in imgs:
            cap_path = os.path.join(self.ner_dir, i + '.pkl')
            cap_ner = pickle.load(open(cap_path, 'rb'))
            cap_text.append(cap_ner)   

            feat_path = os.path.join(self.img_feats_dir, i + '.npy')
            feat = torch.from_numpy(np.load(feat_path))
            combined_feats.append(feat.unsqueeze(0))

            cap_idx = self.caps2id[i]
            cap = self.real_caps[cap_idx]
            tmp = self.preprocess(cap)
            combined_caps.append(tmp)

        num_imgs = len(imgs)
        if num_imgs < 3:
            for r in range(3 - num_imgs):
                combined_feats.append(torch.zeros(1, 36, 2048))
                combined_caps.append(combined_caps[0])
                cap_text.append(set())
        img_exists = torch.zeros(3, dtype=torch.bool)
        for i in range(num_imgs):
            img_exists[i] = True
        combined_feats = torch.cat(combined_feats, dim=0)
        
        combine = list(art)
        combine.append(combined_feats)
        combine.append(img_exists)
        combine.append(combined_caps)
        combine.append(torch.tensor(label, dtype=torch.bool))
        combine.append(art_ner)
        combine.append(cap_text)
        combine = tuple(combine)

        return combine

    def __len__(self):
        return len(self.arts)
