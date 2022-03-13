path2dataset='../data/2022'
wav2vec_small_path='../voiceMOS2022/fairseq/wav2vec_small.pt'
python mos_fairseq.py --datadir "${path2dataset}/phase1-main/DATA/" --fairseq_base_model $wav2vec_small_path --outdir checkpoints_${@: -1}
python predict.py --datadir "${path2dataset}/phase1-main/DATA/" --fairseq_base_model $wav2vec_small_path --finetuned_checkpoint checkpoints_${@: -1}/best.ckpt  --outfile checkpoints_${@: -1}/answer.txt
python mos_fairseq.py --datadir "${path2dataset}/phase1-ood/DATA/" --fairseq_base_model $wav2vec_small_path --finetune_from_checkpoint checkpoints_${@: -1}/best.ckpt --outdir checkpoints_ood_${@: -1}
python predict.py --datadir "${path2dataset}/phase1-ood/DATA/" --fairseq_base_model $wav2vec_small_path --finetuned_checkpoint checkpoints_ood_${@: -1}/best.ckpt --outfile checkpoints_ood_${@: -1}/answer.txt
