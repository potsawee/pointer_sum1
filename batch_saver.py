import os
import sys
import pickle
from tqdm import tqdm

from data_util import config
from data_util.data import Vocab
from data_util.batcher import Batcher

TEST_DATA_SIZE = 11490

def load_batches_decode():

    vocab   = Vocab(config.vocab_path, config.vocab_size)
    batcher = Batcher(config.decode_data_path, vocab, mode='decode',
                           batch_size=config.beam_size, single_pass=True)

    batches = [None for _ in range(TEST_DATA_SIZE)]
    for i in range(TEST_DATA_SIZE):
        batch = batcher.next_batch()
        batches[i] = batch

    with open("lib/data/batches_test.vocab{}.beam{}.pk.bin".format(vocab.size(), config.beam_size), "wb") as f:
        pickle.dump(batches, f)

def load_batches_train():

    vocab   = Vocab(config.vocab_path, config.vocab_size)
    batcher = Batcher(config.decode_data_path, vocab, mode='train',
                           batch_size=config.batch_size, single_pass=False)

    TRAIN_DATA_SIZE = 287226
    num_batches = int(TRAIN_DATA_SIZE / config.batch_size)
    batches = [None for _ in range(num_batches)]
    for i in tqdm(range(num_batches)):
        batch = batcher.next_batch()
        batches[i] = batch

    with open("lib/data/batches_train.vocab{}.batch{}.pk.bin".format(vocab.size(), config.batch_size), "wb") as f:
        pickle.dump(batches, f)

# load_batches_decode()
load_batches_train()
