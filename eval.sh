
MODE=test

VBS=9
BS=8

CHECKPOINT_DIR=/home/andy/Dropbox/largefiles1/logs/starganv2_bs8/checkpoints
TRAINDIR=/home/andy/Dropbox/largefiles1/autoferry_processed/autoferry/train
VALDIR="/home/andy/Dropbox/largefiles1/autoferry_processed/autoferry/val"
EVAL_DIR=inferences

python3 main.py \
--mode $MODE \
--num_domains 2 \
--w_hpf 0 \
--lambda_reg 1 \
--lambda_sty 1 \
--lambda_ds 1 \
--lambda_cyc 1 \
--train_img_dir $TRAINDIR \
--val_img_dir $VALDIR  \
--checkpoint_dir $CHECKPOINT_DIR \
--eval_dir $EVAL_DIR \
--batch_size $BS \
--val_batch_size $VBS \
--img_size 256