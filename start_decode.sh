#!/bin/bash
#$ -S /bin/bash
unset LD_PRELOAD # when run on the stack it has /usr/local/grid/agile/admin/cudadevice.so which will give core dumped
export PATH=/home/mifs/pm574/anaconda3/bin/:$PATH # to use activate
source activate torch11-cuda9
export PYTHONBIN=/home/mifs/pm574/anaconda3/envs/torch11-cuda9/bin/python


export PYTHONPATH=`pwd`
MODEL=$1
START_ID=$2
STOP_ID=$3
export CUDA_VISIBLE_DEVICES=1

# python training_ptr_gen/decode.py $MODEL >& ../log/decode_log &
$PYTHONBIN training_ptr_gen/decode.py $MODEL $START_ID $STOP_ID
