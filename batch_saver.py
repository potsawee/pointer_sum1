import os
import sys
import pickle

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

load_batches_decode()
