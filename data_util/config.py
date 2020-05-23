import os

root_dir = os.path.expanduser("~")

#train_data_path = os.path.join(root_dir, "ptr_nw/cnn-dailymail-master/finished_files/train.bin")
# train_data_path = os.path.join(root_dir, "/home/alta/summary/pm574/data/cnn_dm/finished_files/chunked/train_*")
# eval_data_path = os.path.join(root_dir, "/home/alta/summary/pm574/data/cnn_dm/finished_files/val.bin")
# decode_data_path = os.path.join(root_dir, "/home/alta/summary/pm574/data/cnn_dm/finished_files/test.bin")
vocab_path = os.path.join(root_dir, "/home/alta/summary/pm574/data/cnn_dm/finished_files/vocab")


model_name = "PGN_AMI_APR21B2"
train_dir  = "lib/trained_models/{}".format(model_name)
decode_dir = "lib/decode_asrx/{}".format(model_name)

model_file_path = None
# model_file_path = "lib/trained_models/PTR_COV2/iter330000.pt"

# Hyperparameters
hidden_dim= 256
emb_dim= 128
batch_size=1
max_enc_steps=12000
max_dec_steps=300
beam_size=10
min_dec_steps=35
vocab_size=50000

lr=0.15
adagrad_init_acc=0.1
rand_unif_init_mag=0.02
trunc_norm_init_std=1e-4
max_grad_norm=2.0

pointer_gen = True
is_coverage = True
cov_loss_wt = 1.0
lr_coverage = 0.15

eps = 1e-12
max_iterations = 330000+100*50
save_every     = 100
eval_every     = 100
stop_after     = 3
print_interval = 1

use_gpu = False
random_seed = 334
print("random_seed =", random_seed)
print("model_name =", model_name)
print("beam_size =",beam_size)
# ---- Hierarchical & Sentence-memory modification ---- #
is_hierarchical = False
