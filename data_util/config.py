import os

root_dir = os.path.expanduser("~")

#train_data_path = os.path.join(root_dir, "ptr_nw/cnn-dailymail-master/finished_files/train.bin")
train_data_path = os.path.join(root_dir, "/home/alta/summary/pm574/data/cnn_dm/finished_files/chunked/train_*")
eval_data_path = os.path.join(root_dir, "/home/alta/summary/pm574/data/cnn_dm/finished_files/val.bin")
decode_data_path = os.path.join(root_dir, "/home/alta/summary/pm574/data/cnn_dm/finished_files/test.bin")
vocab_path = os.path.join(root_dir, "/home/alta/summary/pm574/data/cnn_dm/finished_files/vocab")

decode_pk_path = "lib/data/batches_test.vocab50000.beam4.pk.bin"

model_name = "PTR_A2"
train_dir  = "lib/trained_models/{}".format(model_name)
decode_dir = "lib/decode/{}".format(model_name)
print_interval = 100

# Hyperparameters
hidden_dim= 256
emb_dim= 128
batch_size=16
max_enc_steps=400
max_dec_steps=100
beam_size=4
min_dec_steps=35
vocab_size=50000

lr=0.15
adagrad_init_acc=0.1
rand_unif_init_mag=0.02
trunc_norm_init_std=1e-4
max_grad_norm=2.0

pointer_gen = True
is_coverage = False
cov_loss_wt = 1.0

eps = 1e-12
max_iterations = 600000

use_gpu=True
lr_coverage=0.15
