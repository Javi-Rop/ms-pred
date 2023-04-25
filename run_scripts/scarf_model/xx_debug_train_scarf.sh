CUDA_VISIBLE_DEVICES=0 python src/ms_pred/scarf_pred/train_gen.py  \
--seed 5 \
--num-workers 0 \
--batch-size 16 \
--max-epochs 100 \
--dataset-name canopus_train_public \
--dataset-labels labels.tsv \
--split-name split_1.tsv \
--learning-rate 0.001 \
--lr-decay-rate 0.8579 \
--hidden-size 128  \
--gnn-layers 5 \
--mlp-layers 2 \
--set-layers 2 \
--dropout 0 \
--gpu \
--debug-overfit \
--save-dir results/debug_overfit_scarf_gen \
--weight-decay 1e-06 \
--loss bce  \
--use-reverse \
