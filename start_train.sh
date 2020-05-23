#!/bin/bash
#$ -S /bin/bash
unset LD_PRELOAD # when run on the stack it has /usr/local/grid/agile/admin/cudadevice.so which will give core dumped

export PATH=/home/mifs/pm574/anaconda3/bin/:$PATH # to use activate
source activate torch12-cuda10-cl
export PYTHONBIN=/home/mifs/pm574/anaconda3/envs/torch12-cuda10-cl/bin/python3


export PYTHONPATH=`pwd`
export CUDA_VISIBLE_DEVICES=3
# python training_ptr_gen/train.py > LOGs/log.txt &
$PYTHONBIN training_ptr_gen/train_ami.py
