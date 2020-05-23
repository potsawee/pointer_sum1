#!/bin/bash
#$ -S /bin/bash
unset LD_PRELOAD # when run on the stack it has /usr/local/grid/agile/admin/cudadevice.so which will give core dumped

export PATH=/home/mifs/pm574/anaconda3/bin/:$PATH # to use activate
source activate torch12-cuda10-cl
export PYTHONBIN=/home/mifs/pm574/anaconda3/envs/torch12-cuda10-cl/bin/python3

export PYTHONPATH=`pwd`
MODEL=$1
START_ID=$2
STOP_ID=$3
export CUDA_VISIBLE_DEVICES=0

# python training_ptr_gen/decode.py $MODEL >& ../log/decode_log &
$PYTHONBIN training_ptr_gen/decode_ami.py $MODEL $START_ID $STOP_ID
