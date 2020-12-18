import copy
import sys
import torch
import torch.nn as nn
import spacy
from pytorch_transformers import BertModel, BertConfig
from torch.nn.init import xavier_uniform_

from models.decoder import TransformerDecoder
from models.encoder import Classifier, ExtTransformerEncoder
from models.optimizers import Optimizer
from models.model_builder import AbsSummarizer

class SCAN(nn.Module):
    def __init__(self, args):
        super(SCAN, self).__init__()
        self.args = args
        self.cos = nn.CosineSimilarity(dim=2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, img_feats, cap_feats, cap_mask):
        #sys.exit()
    
        img_feats = torch.reshape(img_feats, (-1, img_feats.size(-2), img_feats.size(-1)))
        combined_scores = []
        for i in range(cap_feats.size(1)):
            tmp_cap = cap_feats[:, i, :].unsqueeze(1).expand_as(img_feats)
            scores = self.cos(img_feats, tmp_cap)
            combined_scores.append(scores.unsqueeze(-1))
        combined_scores = torch.cat(combined_scores, dim=-1)
        combined_scores = self.softmax(combined_scores)
        
        img_feats = img_feats.unsqueeze(2).repeat(1, 1, combined_scores.size(2), 1)
        combined_scores = combined_scores.unsqueeze(-1).repeat(1, 1, 1, img_feats.size(-1))
        img_cap_reps = combined_scores * img_feats
        img_cap_reps = torch.sum(img_cap_reps, dim=1)
   
        return img_cap_reps

class Model(nn.Module):
    def __init__(self, args, device, checkpoint=None, bert_from_extractive=None):
        super(Model, self).__init__()
        self.args = args
        self.num_imgs = args.num_imgs
        self.encoder = AbsSummarizer(args, device, None, bert_from_extractive)
        self.vis_fc = nn.Sequential(nn.Linear(self.args.img_feat_size, self.args.enc_hidden_size * 2), nn.ReLU())
        self.art_fc = nn.Sequential(nn.Linear(self.args.dec_hidden_size, self.args.enc_hidden_size * 2), nn.ReLU())
        self.cap_fc = nn.Sequential(nn.Linear(self.args.dec_hidden_size, self.args.enc_hidden_size * 2), nn.ReLU())
        self.cls_fc = nn.Sequential(nn.Linear(self.args.enc_hidden_size*4+1, self.args.enc_hidden_size*2), nn.ReLU(), nn.BatchNorm1d(self.args.enc_hidden_size*2), nn.Linear(self.args.enc_hidden_size*2, self.args.enc_hidden_size), nn.ReLU(), nn.Linear(self.args.enc_hidden_size, 2))
        self.scan = SCAN(args)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        self.loss = nn.BCELoss()

    def forward(self, data):
        src, tgt, segs, clss, mask_src, mask_tgt, mask_cls, img_feats, img_exists, cap_src, cap_tgt, cap_segs, cap_clss, cap_mask_src, cap_mask_tgt, cap_mask_cls, labels, art_ner, cap_ner = data

        word_embeds, _, _ = self.encoder(src.cuda(), tgt.cuda(), segs.cuda(), clss.cuda(), mask_src.cuda(), mask_tgt.cuda(), mask_cls.cuda())
        num_toks = torch.sum(mask_src, dim=-1)
        art_embeds = torch.sum(word_embeds, dim=1)
        art_embeds = art_embeds / num_toks.unsqueeze(-1).expand_as(art_embeds).cuda().float()

        cap_word_embeds, _, _ = self.encoder(cap_src.cuda(), cap_tgt.cuda(), cap_segs.cuda(), cap_clss.cuda(), cap_mask_src.cuda(), cap_mask_tgt.cuda(), cap_mask_cls.cuda())
        cap_num_toks = torch.sum(cap_mask_src, dim=-1)
        cap_word_embeds = self.cap_fc(cap_word_embeds)
        img_feats = self.vis_fc(img_feats.cuda())
        
        img_cap_reps = self.scan(img_feats, cap_word_embeds, cap_mask_src)
        img_cap_reps = torch.sum(img_cap_reps, dim=1)    

        cap_tok_counts = torch.sum(cap_mask_src, dim=-1)
        cap_tok_counts = cap_tok_counts.unsqueeze(-1).expand_as(img_cap_reps)
        img_cap_reps = img_cap_reps / cap_tok_counts.float().cuda()

        art_embeds = self.art_fc(art_embeds)

        if self.training:
            loss = 0.
            art_scores = []
            img_cap_reps = torch.reshape(img_cap_reps, (len(art_embeds), self.num_imgs, -1))
            for i in range(len(art_embeds)):   

                tmp_art_text = art_ner[i]
                curr_cap = []
                for j in cap_ner:
                    for k in j:
                        overlap = tmp_art_text.intersection(k)
                        if len(overlap) >= 1:
                            curr_cap.append(torch.tensor(1.).unsqueeze(0))
                        else:
                            curr_cap.append(torch.tensor(0.).unsqueeze(0))
                curr_cap = torch.cat(curr_cap, dim=0)
                curr_cap = torch.reshape(curr_cap, (-1, self.num_imgs))
                curr_cap = curr_cap.unsqueeze(-1)
             
                tmp_art = art_embeds[i].unsqueeze(0).unsqueeze(0).expand_as(img_cap_reps)
                tmp_reps = torch.cat((tmp_art, img_cap_reps, curr_cap.cuda()), dim=-1)

                tmp_reps = torch.reshape(tmp_reps, (-1, tmp_reps.size(-1)))

                tmp_scores = self.cls_fc(tmp_reps)
                tmp_scores = self.sigmoid(tmp_scores)
                tmp_scores = torch.reshape(tmp_scores, (len(art_embeds), self.num_imgs, tmp_scores.size(-1)))
                tmp_exists = img_exists.unsqueeze(-1).expand_as(tmp_scores)
                tmp_scores = tmp_scores.masked_fill(tmp_exists.cuda() == 0, 0.)
                tmp_scores = 1. - tmp_scores
                tmp_scores = torch.prod(tmp_scores, 1)
                tmp_scores = 1. - tmp_scores
                tmp_labels = torch.zeros(len(art_embeds), dtype=torch.bool)
                tmp_labels[i] = labels[i]
                tmp_inv = ~tmp_labels
                fake_loss = tmp_scores[:, 0]
                fake_loss = self.loss(fake_loss, tmp_inv.float().cuda())
                real_loss = tmp_scores[:, 1]
                real_loss = self.loss(real_loss, tmp_labels.float().cuda())
                tmp_loss = fake_loss + real_loss
                tmp_loss /= len(art_embeds)
                loss += tmp_loss
                art_scores.append(tmp_scores[i].unsqueeze(0))
            
            art_scores = torch.cat(art_scores, dim=0)
            return loss, art_scores

        curr_cap = []
        for i in range(len(art_ner)):
            tmp_art = art_ner[i]
            tmp_cap = cap_ner[i]
            for j in tmp_cap:
                overlap = tmp_art.intersection(j)
                if len(overlap) >= 1:
                    curr_cap.append(torch.tensor(1.).unsqueeze(0))
                else:
                    curr_cap.append(torch.tensor(0.).unsqueeze(0))
        curr_cap = torch.cat(curr_cap, dim=0)
        curr_cap = curr_cap.unsqueeze(-1)

        art_embeds = art_embeds.repeat(self.num_imgs, 1)

        art_cls_reps = torch.cat((art_embeds, img_cap_reps, curr_cap.cuda()), dim=-1)

        art_scores = self.cls_fc(art_cls_reps)
        art_scores = self.sigmoid(art_scores).squeeze()

        img_exists = torch.reshape(img_exists, (-1, 1)).squeeze().cuda()
        img_exists = img_exists.unsqueeze(-1).expand_as(art_scores)
        art_scores = art_scores.masked_fill(img_exists == 0, 0.)

        art_scores = 1. - torch.reshape(art_scores, shape=(-1, self.num_imgs, art_scores.size(-1)))
        art_scores = torch.prod(art_scores, 1)
        art_scores = 1. - art_scores
        
        return art_scores
