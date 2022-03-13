# ==============================================================================
# Copyright (c) 2021, Yamagishi Laboratory, National Institute of Informatics
# Author: Erica Cooper
# All rights reserved.
# ==============================================================================

import os
import argparse
import fairseq
import torch
import torchaudio
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import random
import scipy.stats
import numpy as np
random.seed(1984)

def systemID(uttID):
    return uttID.split('-')[0]
def get_true_sys_mos(system_csv_path):
    ### SYSTEM
    true_sys_MOS_avg = { }
    csv_file = open(system_csv_path, 'r')
    csv_file.readline()  ## skip header
    for line in csv_file:
        parts = line.strip().split(',')
        sysID = parts[0]
        MOS = float(parts[1])
        true_sys_MOS_avg[sysID] = MOS
    return true_sys_MOS_avg

def compute_sys_metrics(predictions, true_sys_MOS_avg):
    sorted_uttIDs = sorted(predictions.keys())
    pred_sys_MOSes = { }
    for uttID in sorted_uttIDs:
        sysID = systemID(uttID)
        noop = pred_sys_MOSes.setdefault(sysID, [ ])
        pred_sys_MOSes[sysID].append(predictions[uttID])

    pred_sys_MOS_avg = { }
    for k, v in pred_sys_MOSes.items():
        avg_MOS = sum(v) / (len(v) * 1.0)
        pred_sys_MOS_avg[k] = avg_MOS
    
    ## make lists sorted by system
    pred_sysIDs = sorted(pred_sys_MOS_avg.keys())
    sys_p = [ ]
    sys_t = [ ]
    for sysID in pred_sysIDs:
        sys_p.append(pred_sys_MOS_avg[sysID])
        sys_t.append(true_sys_MOS_avg[sysID])

    sys_true = np.array(sys_t)
    sys_predicted = np.array(sys_p)
    
    MSE=np.mean((sys_true-sys_predicted)**2)
    LCC=np.corrcoef(sys_true, sys_predicted)
    SRCC=scipy.stats.spearmanr(sys_true.T, sys_predicted.T)
    KTAU=scipy.stats.kendalltau(sys_true, sys_predicted)
    return dict(SRCC=SRCC[0], MSE=MSE, LCC=LCC[0][1], KTAU=KTAU[0])

def compute_utt_metrics(predictions, true_MOS):
    ts = []
    ps = []
    sorted_uttIDs = sorted(predictions.keys())
    for uttID in sorted_uttIDs:
        t = true_MOS[uttID]
        p = predictions[uttID]
        ts.append(t)
        ps.append(p)
    truths = np.array(ts)
    preds = np.array(ps)
    
    
    MSE=np.mean((truths-preds)**2)
    LCC=np.corrcoef(truths, preds)
    SRCC=scipy.stats.spearmanr(truths.T, preds.T)
    KTAU=scipy.stats.kendalltau(truths, preds)

    return dict(MSE=MSE, LCC=LCC[0][1], SRCC=SRCC[0], KTAU=KTAU[0])

class MosPredictor(nn.Module):
    def __init__(self, ssl_model, ssl_out_dim):
        super(MosPredictor, self).__init__()
        self.ssl_model = ssl_model
        self.ssl_features = ssl_out_dim
        self.output_layer = nn.Linear(self.ssl_features, 1)
        
    def forward(self, wav):
        wav = wav.squeeze(1)  ## [batches, audio_len]
        res = self.ssl_model(wav, mask=False, features_only=True)
        x = res['x']
        x = torch.mean(x, 1)
        x = self.output_layer(x)
        return x.squeeze(1)

    
class MyDataset(Dataset):
    def __init__(self, wavdir, mos_list):
        self.mos_lookup = { }
        f = open(mos_list, 'r')
        for line in f:
            parts = line.strip().split(',')
            wavname = parts[0]
            mos = float(parts[1])
            self.mos_lookup[wavname] = mos

        self.wavdir = wavdir
        self.wavnames = sorted(self.mos_lookup.keys())

        
    def __getitem__(self, idx):
        wavname = self.wavnames[idx]
        wavpath = os.path.join(self.wavdir, wavname)
        wav = torchaudio.load(wavpath)[0]
        score = self.mos_lookup[wavname]
        return wav, score, wavname
    

    def __len__(self):
        return len(self.wavnames)


    def collate_fn(self, batch):  ## zero padding
        wavs, scores, wavnames = zip(*batch)
        wavs = list(wavs)
        max_len = max(wavs, key = lambda x : x.shape[1]).shape[1]
        output_wavs = []
        for wav in wavs:
            amount_to_pad = max_len - wav.shape[1]
            padded_wav = torch.nn.functional.pad(wav, (0, amount_to_pad), 'constant', 0)
            output_wavs.append(padded_wav)

        output_wavs = torch.stack(output_wavs, dim=0)
        scores  = torch.stack([torch.tensor(x) for x in list(scores)], dim=0)
        return output_wavs, scores, wavnames

    
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, required=True, help='Path of your DATA/ directory')
    parser.add_argument('--fairseq_base_model', type=str, required=True, help='Path to pretrained fairseq base model')
    parser.add_argument('--finetune_from_checkpoint', type=str, required=False, help='Path to your checkpoint to finetune from')
    parser.add_argument('--outdir', type=str, required=False, default='checkpoints', help='Output directory for your trained checkpoints')
    args = parser.parse_args()

    cp_path = args.fairseq_base_model
    datadir = args.datadir
    ckptdir = args.outdir
    my_checkpoint = args.finetune_from_checkpoint
    
    if not os.path.exists(ckptdir):
        os.system('mkdir -p ' + ckptdir)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('DEVICE: ' + str(device))

    wavdir = os.path.join(datadir, 'wav')
    trainlist = os.path.join(datadir, 'sets/train_mos_list.txt')
    validlist = os.path.join(datadir, 'sets/val_mos_list.txt')

    ssl_model_type = cp_path.split('/')[-1]
    if ssl_model_type == 'wav2vec_small.pt':
        SSL_OUT_DIM = 768
    elif ssl_model_type in ['w2v_large_lv_fsh_swbd_cv.pt', 'xlsr_53_56k.pt']:
        SSL_OUT_DIM = 1024
    else:
        print('*** ERROR *** SSL model type ' + ssl_model_type + ' not supported.')
        exit()

    model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
    ssl_model = model[0]
    ssl_model.remove_pretraining_modules()
    
    trainset = MyDataset(wavdir, trainlist)
    trainloader = DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2, collate_fn=trainset.collate_fn)

    validset = MyDataset(wavdir, validlist)
    validloader = DataLoader(validset, batch_size=1, shuffle=False, num_workers=2, collate_fn=validset.collate_fn)

    net = MosPredictor(ssl_model, SSL_OUT_DIM)
    net = net.to(device)

    if my_checkpoint != None:  ## do (further) finetuning
        net.load_state_dict(torch.load(my_checkpoint))
    
    criterion = nn.L1Loss()
    optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)

    PREV_VAL_LOSS=9999999999
    orig_patience=20
    patience=orig_patience
    for epoch in range(1,1001):
        STEPS=0
        net.train()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels, filenames = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            STEPS += 1
            running_loss += loss.item()
        print('EPOCH: ' + str(epoch))
        print('AVG EPOCH TRAIN LOSS: ' + str(running_loss / STEPS))
        epoch_val_loss = 0.0
        net.eval()
        ## clear memory to avoid OOM
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
        ## validation
        VALSTEPS=0
        predictions = {}
        with torch.no_grad():
            for i, data in enumerate(validloader, 0):
                VALSTEPS+=1
                inputs, labels, filenames = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                epoch_val_loss += loss.item()
                for filename, utt_score in zip(filenames,outputs):
                    predictions[filename] = utt_score.detach().cpu().numpy()

        val_metrics = compute_sys_metrics(predictions, get_true_sys_mos(os.path.join(datadir,"mydata_system.csv")))
        val_srcc = -val_metrics['SRCC']
        avg_val_loss=epoch_val_loss/VALSTEPS    
        for k, v in val_metrics.items():
            print(f'EPOCH sysytem {k}: {v}')
        print('EPOCH VAL LOSS: ' + str(avg_val_loss))
        print('EPOCH SRCC: ' + str(-val_srcc))
        if val_srcc < PREV_VAL_LOSS:
            print('SRCC has increased')
            PREV_VAL_LOSS=val_srcc
            PATH = os.path.join(ckptdir, "best.ckpt")
            torch.save(net.state_dict(), PATH)
            patience = orig_patience
        else:
            patience-=1
            if patience == 0:
                print('loss has not decreased for ' + str(orig_patience) + ' epochs; early stopping at epoch ' + str(epoch))
                break
        
    print('Finished Training')

if __name__ == '__main__':
    main()
