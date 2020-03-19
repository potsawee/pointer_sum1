export PYTHONPATH=`pwd`
MODEL_PATH=$1
MODEL_NAME=$(basename $MODEL_PATH)
export CUDA_VISIBLE_DEVICES=0

# python training_ptr_gen/eval.py $MODEL_PATH >& ../log/eval_log.$MODEL_NAME &
python training_ptr_gen/eval.py $MODEL_PATH
