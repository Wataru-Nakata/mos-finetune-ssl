# ==============================================================================
# Copyright (c) 2021, Yamagishi Laboratory, National Institute of Informatics
# Author: Erica Cooper
# All rights reserved.
# ==============================================================================

import os
import argparse
import torch
import torch.nn as nn
import fairseq
from torch.utils.data import DataLoader
from mos_fairseq import MosPredictor, MyDataset
import numpy as np
import scipy.stats


def systemID(uttID):
    return uttID.split('-')[0]


import os.path

import numpy as np
import scipy
import scipy.stats
import csv

MAX_SCORE=100
MIN_SCORE=0

def calculate_scores(truth, submission_answer, prefix):
    # sanity check
    sanity_ok = True
    diff = []
    for wav_name in truth:
        if wav_name not in submission_answer:
            diff.append(wav_name)
            sanity_ok = False
    if not sanity_ok:
        print("Sanity check for {} track failed. These files are not in submission:".format(prefix))
        return {
            prefix + "_UTT_MSE": MAX_SCORE,
            prefix + "_UTT_LCC": MIN_SCORE,
            prefix + "_UTT_SRCC": MIN_SCORE,
            prefix + "_UTT_KTAU": MIN_SCORE,
            prefix + "_SYS_MSE": MAX_SCORE,
            prefix + "_SYS_LCC": MIN_SCORE,
            prefix + "_SYS_SRCC": MIN_SCORE,
            prefix + "_SYS_KTAU": MIN_SCORE,
        }
    else:
        print("Sanity check for {} track succeeded.".format(prefix))
    
    # utterance level scores
    sorted_truth = np.array([truth[k] for k in sorted(truth)])
    sorted_submission_answer = np.array([submission_answer[k] for k in sorted(submission_answer) if k in truth])
    UTT_MSE=np.mean((sorted_truth-sorted_submission_answer)**2)
    UTT_LCC=np.corrcoef(sorted_truth, sorted_submission_answer)[0][1]
    UTT_SRCC=scipy.stats.spearmanr(sorted_truth, sorted_submission_answer)[0]
    UTT_KTAU=scipy.stats.kendalltau(sorted_truth, sorted_submission_answer)[0]

    # system level scores
    sorted_system_list = sorted(list(set([k.split("-")[0] for k in truth.keys()])))
    sys_truth = {system: [v for k, v in truth.items() if k.startswith(system)] for system in sorted_system_list}
    sys_submission = {system: [v for k, v in submission_answer.items() if k.startswith(system)] for system in sorted_system_list}
    sorted_sys_truth = np.array([np.mean(group) for group in sys_truth.values()])
    sorted_sys_submission = np.array([np.mean(group) for group in sys_submission.values()])
    SYS_MSE=np.mean((sorted_sys_truth-sorted_sys_submission)**2)
    SYS_LCC=np.corrcoef(sorted_sys_truth, sorted_sys_submission)[0][1]
    SYS_SRCC=scipy.stats.spearmanr(sorted_sys_truth, sorted_sys_submission)[0]
    SYS_KTAU=scipy.stats.kendalltau(sorted_sys_truth, sorted_sys_submission)[0]

    return {
        prefix + "_UTT_MSE": UTT_MSE,
        prefix + "_UTT_LCC": UTT_LCC,
        prefix + "_UTT_SRCC": UTT_SRCC,
        prefix + "_UTT_KTAU": UTT_KTAU,
        prefix + "_SYS_MSE": SYS_MSE,
        prefix + "_SYS_LCC": SYS_LCC,
        prefix + "_SYS_SRCC": SYS_SRCC,
        prefix + "_SYS_KTAU": SYS_KTAU,
    }
def read_file(filepath):
    with open(filepath, "r") as csvfile:
        rows = list(csv.reader(csvfile))
    return {os.path.splitext(row[0])[0]: float(row[1]) for row in rows}
# open the truth file

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fairseq_base_model', type=str, required=True, help='Path to pretrained fairseq base model.')
    parser.add_argument('--datadir', type=str, required=True, help='Path of your DATA/ directory')
    parser.add_argument('--finetuned_checkpoint', type=str, required=True, help='Path to finetuned MOS prediction checkpoint.')
    parser.add_argument('--outfile', type=str, required=False, default='answer.txt', help='Output filename for your answer.txt file for submission to the CodaLab leaderboard.')
    args = parser.parse_args()
    
    cp_path = args.fairseq_base_model
    my_checkpoint = args.finetuned_checkpoint
    datadir = args.datadir
    outfile = args.outfile


    model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
    ssl_model = model[0]
    ssl_model.remove_pretraining_modules()

    print('Loading checkpoint')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ssl_model_type = cp_path.split('/')[-1]
    if ssl_model_type == 'wav2vec_small.pt':
        SSL_OUT_DIM = 768
    elif ssl_model_type in ['w2v_large_lv_fsh_swbd_cv.pt', 'xlsr_53_56k.pt']:
        SSL_OUT_DIM = 1024
    else:
        print('*** ERROR *** SSL model type ' + ssl_model_type + ' not supported.')
        exit()

    model = MosPredictor(ssl_model, SSL_OUT_DIM).to(device)
    model.eval()

    model.load_state_dict(torch.load(my_checkpoint))

    wavdir = os.path.join(datadir, 'wav')
    validlist = os.path.join(datadir, 'sets/val_mos_list.txt')
    testlist = os.path.join(datadir, 'sets/test_mos_list.txt')

    print('Loading data')
    validset = MyDataset(wavdir, validlist)
    testset = MyDataset(wavdir, testlist)
    validloader = DataLoader(validset, batch_size=1, shuffle=True, num_workers=2, collate_fn=validset.collate_fn)
    testloader = DataLoader(testset, batch_size=1, shuffle=True, num_workers=2, collate_fn=validset.collate_fn)

    total_loss = 0.0
    num_steps = 0.0
    predictions = { }  # filename : prediction
    criterion = nn.L1Loss()
    print('Starting prediction')

    for i, data in enumerate(validloader, 0):
        inputs, labels, filenames = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        
        output = outputs.cpu().detach().numpy()[0]
        predictions[filenames[0]] = output  ## batch size = 1

    bvcc_truth_file = os.path.join(datadir, "sets/test_mos_list.txt")
    bvcc_truth = read_file(bvcc_truth_file)
    bvcc_truth_file_dev = os.path.join(datadir, "sets/val_mos_list.txt")
    bvcc_truth_dev = read_file(bvcc_truth_file_dev)
    calculate_scores(bvcc_truth_dev,predictions,prefix=datadir.split('/')[-1]+"DEV")
    ans = open(outfile+'_dev', 'w')
    for k, v in predictions.items():
        outl = k.split('.')[0] + ',' + str(v) + '\n'
        ans.write(outl)
    ans.close()
    for i, data in enumerate(testloader, 0):
        inputs, labels, filenames = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        
        output = outputs.cpu().detach().numpy()[0]
        predictions[filenames[0]] = output  ## batch size = 1

    ans = open(outfile+"_test", 'w')
    for k, v in predictions.items():
        outl = k.split('.')[0] + ',' + str(v) + '\n'
        ans.write(outl)
    ans.close()
    calculate_scores(bvcc_truth,predictions,prefix=datadir.split('/')[-1]+"TEST")

if __name__ == '__main__':
    main()
