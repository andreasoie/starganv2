
MODE=cherry

VBS=1
BS=1

EVAL_DIR=cherries_reference

CHECKPOINT=100000
CHECKPOINT_DIR=/home/andreoi/ckpts/autoferry_starganv2/checkpoints
TRAINDIR=/home/andreoi/data/study_cases/train
VALDIR=/home/andreoi/data/study_cases

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
--resume_iter $CHECKPOINT \
--eval_dir $EVAL_DIR \
--batch_size $BS \
--val_batch_size $VBS \
--img_size 256