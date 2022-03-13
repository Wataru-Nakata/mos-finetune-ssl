path2dataset='../data/2022/'
wav2vec_small_path='fairseq/wav2vec_small.pt'
python mos_fairseq.py --datadir $path2datset/phase1-main/DATA/ --fairseq_base_model fairseq/wav2vec_small.pt --outdir checkpoints_${@: -1}
python predict.py --datadir $path2datset/phase1-main/DATA/ --fairseq_base_model fairseq/wav2vec_small.pt --finetuned_checkpoint checkpoints_${@: -1}/best.ckpt 
python mos_fairseq.py --datadir $path2datset/phase1-ood/DATA/ --fairseq_base_model fairseq/wav2vec_small.pt --finetuned_checkpoint checkpoints_${@: -1}/best.ckpt --outdir checkpoints_ood_${@: -1}
python predict.py --datadir $path2datset/phase1-ood/DATA/ --fairseq_base_model fairseq/wav2vec_small.pt --finetuned_checkpoint checkpoints_ood_${@:-1/best.ckpt --outdir checkpoints_ood_${@: -1}
