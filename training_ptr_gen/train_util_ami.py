# import sys
# sys.path.insert(0, '/home/alta/summary/pm574/pointer_summarizer')

from torch.autograd import Variable
import numpy as np
import torch
import pickle

from data_util import config
from data_util import data
from data_util.data import Vocab

from nltk import word_tokenize
from transformers import BertTokenizer
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

class TopicSegment(object):
    def __init__(self):
        self.utterances = []
        self.topic_labels = None
        self.topic_description = None

    def add_utterance(self, utterance):
        self.utterances.append(utterance)

class Utterance(object):
    def __init__(self, encoded_words, dialogueact, speakerid, extsum_label):
        self.encoded_words = encoded_words
        self.dialogueact   = dialogueact
        self.speakerid     = speakerid
        self.extsum_label  = extsum_label

def load_ami_data(data_type):
    path = "/home/alta/summary/pm574/summariser1/lib/model_data/ami-191209.{}.pk.bin".format(data_type)
    with open(path, 'rb') as f:
        ami_data = pickle.load(f, encoding="bytes")
    return ami_data

# vocab = Vocab(config.vocab_path, config.vocab_size)
# PAD_ID   = vocab.word2id(data.PAD_TOKEN)
# START_ID = vocab.word2id(data.START_DECODING)
# STOP_ID  = vocab.word2id(data.STOP_DECODING)

class Example(object):
    def __init__(self, enc_input, enc_len, dec_input, dec_len, target,
                enc_input_extend_vocab, article_oovs):

        self.enc_input = enc_input
        self.enc_len   = enc_len
        self.dec_input = dec_input
        self.dec_len   = dec_len
        self.target    = target
        self.enc_input_extend_vocab = enc_input_extend_vocab
        self.article_oovs = article_oovs

    def pad_encoder_input(self, max_len, pad_id):
        while len(self.enc_input) < max_len:
            self.enc_input.append(pad_id)
        if config.pointer_gen:
            while len(self.enc_input_extend_vocab) < max_len:
                self.enc_input_extend_vocab.append(pad_id)

    def pad_decoder_inp_targ(self, max_len, pad_id):
        while len(self.dec_input) < max_len:
            self.dec_input.append(pad_id)
        while len(self.target) < max_len:
            self.target.append(pad_id)

def get_a_batch(ami_data, idx, vocab, batch_size, max_enc_steps, max_dec_steps,
                start_id, stop_id, pad_id, sum_type, use_cuda):
    if sum_type not in ['long', 'short']:
        raise Exception("summary type long/short only")

    example_list = [None for _ in range(batch_size)]

    for bn in range(batch_size):
        topic_segments  = ami_data[idx+bn][0]
        if sum_type == 'long':    encoded_summary = ami_data[idx+bn][1]
        elif sum_type == 'short': encoded_summary = ami_data[idx+bn][2]
        # input
        meeting_words = []
        for segment in topic_segments:
            utterances = segment.utterances
            for utterance in utterances:
                encoded_words = utterance.encoded_words
                meeting_words += encoded_words

        meeting_word_string = bert_tokenizer.decode(meeting_words)
        # summary
        summary_string = bert_tokenizer.decode(encoded_summary)
        summary_string = summary_string.replace('[CLS]', '')
        summary_string = summary_string.replace('[MASK]', '')
        summary_string = summary_string.replace('[SEP]', '')

        # create an example
        article_words = word_tokenize(meeting_word_string)
        if len(article_words) > config.max_enc_steps:
            article_words = article_words[:config.max_enc_steps]
        enc_len = len(article_words)
        enc_input = [vocab.word2id(w) for w in article_words]

        abstract_words = word_tokenize(summary_string)
        abs_ids = [vocab.word2id(w) for w in abstract_words]

        dec_input, target = get_dec_inp_targ_seqs(abs_ids, config.max_dec_steps, start_id, stop_id)
        dec_len = len(dec_input)

        # If using pointer-generator mode, we need to store some extra info
        if config.pointer_gen:
            # Store a version of the enc_input where in-article OOVs are represented by their temporary OOV id; also store the in-article OOVs words themselves
            enc_input_extend_vocab, article_oovs = data.article2ids(article_words, vocab)

            # Get a verison of the reference summary where in-article OOVs are represented by their temporary article OOV id
            abs_ids_extend_vocab = data.abstract2ids(abstract_words, vocab, article_oovs)

            # Overwrite decoder target sequence so it uses the temp article OOV ids
            _, target = get_dec_inp_targ_seqs(abs_ids_extend_vocab, config.max_dec_steps, start_id, stop_id)
        else:
            enc_input_extend_vocab = None
            article_oovs = None

        example = Example(enc_input, enc_len, dec_input, dec_len, target,
                         enc_input_extend_vocab, article_oovs)

        example_list[bn] = example

    ###################### init encoder seq ######################
    max_enc_seq_len = max([ex.enc_len for ex in example_list])
    for ex in example_list:
        ex.pad_encoder_input(max_enc_seq_len, pad_id)

    # Initialize the numpy arrays
    # Note: our enc_batch can have different length (second dimension) for each batch because we use dynamic_rnn for the encoder.
    enc_batch = np.zeros((batch_size, max_enc_seq_len), dtype=np.int32)
    enc_lens = np.zeros((batch_size), dtype=np.int32)
    enc_padding_mask = np.zeros((batch_size, max_enc_seq_len), dtype=np.float32)
    # Fill in the numpy arrays
    for i, ex in enumerate(example_list):
        enc_batch[i, :] = ex.enc_input[:]
        enc_lens[i] = ex.enc_len
        for j in range(ex.enc_len):
            enc_padding_mask[i][j] = 1

    # For pointer-generator mode, need to store some extra info
    if config.pointer_gen:
        # Determine the max number of in-article OOVs in this batch
        max_art_oovs = max([len(ex.article_oovs) for ex in example_list])
        # Store the in-article OOVs themselves
        art_oovs = [ex.article_oovs for ex in example_list]
        # Store the version of the enc_batch that uses the article OOV ids
        enc_batch_extend_vocab = np.zeros((batch_size, max_enc_seq_len), dtype=np.int32)
        for i, ex in enumerate(example_list):
            enc_batch_extend_vocab[i, :] = ex.enc_input_extend_vocab[:]

    ###################### init decoder seq ######################
    # Pad the inputs and targets
    for ex in example_list:
        ex.pad_decoder_inp_targ(config.max_dec_steps, pad_id)

    # Initialize the numpy arrays.
    dec_batch = np.zeros((batch_size, config.max_dec_steps), dtype=np.int32)
    target_batch = np.zeros((batch_size, config.max_dec_steps), dtype=np.int32)
    dec_padding_mask = np.zeros((batch_size, config.max_dec_steps), dtype=np.float32)
    dec_lens = np.zeros((batch_size), dtype=np.int32)

    # Fill in the numpy arrays
    for i, ex in enumerate(example_list):
        dec_batch[i, :] = ex.dec_input[:]
        target_batch[i, :] = ex.target[:]
        dec_lens[i] = ex.dec_len
        for j in range(ex.dec_len):
            dec_padding_mask[i][j] = 1

    # ------------------------------------------------------------------------------- #
    # ---------------- get_input_from_batch , get_output_from_batch ----------------- #
    # ------------------------------------------------------------------------------- #
    # get_input_from_batch
    enc_batch = Variable(torch.from_numpy(enc_batch).long())
    enc_padding_mask = Variable(torch.from_numpy(enc_padding_mask)).float()

    if config.pointer_gen:
        enc_batch_extend_vocab = Variable(torch.from_numpy(enc_batch_extend_vocab).long())
        # max_art_oovs is the max over all the article oov list in the batch
        if max_art_oovs > 0:
            extra_zeros = Variable(torch.zeros((batch_size, max_art_oovs)))

    c_t_1 = Variable(torch.zeros((batch_size, 2 * config.hidden_dim)))

    coverage = None
    if config.is_coverage:
        coverage = Variable(torch.zeros(enc_batch.size()))

    if use_cuda:
        enc_batch = enc_batch.cuda()
        enc_padding_mask = enc_padding_mask.cuda()

        if enc_batch_extend_vocab is not None:
            enc_batch_extend_vocab = enc_batch_extend_vocab.cuda()
        if extra_zeros is not None:
            extra_zeros = extra_zeros.cuda()
        c_t_1 = c_t_1.cuda()

        if coverage is not None:
            coverage = coverage.cuda()

    # get_output_from_batch
    dec_batch = Variable(torch.from_numpy(dec_batch).long())
    dec_padding_mask = Variable(torch.from_numpy(dec_padding_mask)).float()
    max_dec_len = np.max(dec_lens)
    dec_lens_var = Variable(torch.from_numpy(dec_lens)).float()
    target_batch = Variable(torch.from_numpy(target_batch)).long()

    if use_cuda:
        dec_batch = dec_batch.cuda()
        dec_padding_mask = dec_padding_mask.cuda()
        dec_lens_var = dec_lens_var.cuda()
        target_batch = target_batch.cuda()

    return (enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_1, coverage), (dec_batch, dec_padding_mask, max_dec_len, dec_lens_var, target_batch)

def get_a_batch_decode(ami_data, idx, vocab, beam_width, max_enc_steps, max_dec_steps,
                      start_id, stop_id, pad_id, sum_type, use_cuda):
    if sum_type not in ['long', 'short']:
        raise Exception("summary type long/short only")

    topic_segments  = ami_data[idx][0]
    if sum_type == 'long':    encoded_summary = ami_data[idx][1]
    elif sum_type == 'short': encoded_summary = ami_data[idx][2]
    # input
    meeting_words = []
    for segment in topic_segments:
        utterances = segment.utterances
        for utterance in utterances:
            encoded_words = utterance.encoded_words
            meeting_words += encoded_words

    meeting_word_string = bert_tokenizer.decode(meeting_words)
    # summary
    summary_string = bert_tokenizer.decode(encoded_summary)
    summary_string = summary_string.replace('[CLS]', '')
    summary_string = summary_string.replace('[MASK]', '')
    summary_string = summary_string.replace('[SEP]', '')

    # create an example
    article_words = word_tokenize(meeting_word_string)
    if len(article_words) > config.max_enc_steps:
        article_words = article_words[:config.max_enc_steps]
    enc_len = len(article_words)
    enc_input = [vocab.word2id(w) for w in article_words]

    abstract_words = word_tokenize(summary_string)
    abs_ids = [vocab.word2id(w) for w in abstract_words]

    dec_input, target = get_dec_inp_targ_seqs(abs_ids, config.max_dec_steps, start_id, stop_id)
    dec_len = len(dec_input)

    # If using pointer-generator mode, we need to store some extra info
    if config.pointer_gen:
        # Store a version of the enc_input where in-article OOVs are represented by their temporary OOV id; also store the in-article OOVs words themselves
        enc_input_extend_vocab, article_oovs = data.article2ids(article_words, vocab)

        # Get a verison of the reference summary where in-article OOVs are represented by their temporary article OOV id
        abs_ids_extend_vocab = data.abstract2ids(abstract_words, vocab, article_oovs)

        # Overwrite decoder target sequence so it uses the temp article OOV ids
        _, target = get_dec_inp_targ_seqs(abs_ids_extend_vocab, config.max_dec_steps, start_id, stop_id)
    else:
        enc_input_extend_vocab = None
        article_oovs = None

    example_list = [Example(enc_input, enc_len, dec_input, dec_len, target,
                    enc_input_extend_vocab, article_oovs) for _ in range(beam_width)]

    ###################### init encoder seq ######################
    max_enc_seq_len = max([ex.enc_len for ex in example_list])
    for ex in example_list:
        ex.pad_encoder_input(max_enc_seq_len, pad_id)

    # Initialize the numpy arrays
    # Note: our enc_batch can have different length (second dimension) for each batch because we use dynamic_rnn for the encoder.
    enc_batch = np.zeros((beam_width, max_enc_seq_len), dtype=np.int32)
    enc_lens = np.zeros((beam_width), dtype=np.int32)
    enc_padding_mask = np.zeros((beam_width, max_enc_seq_len), dtype=np.float32)
    # Fill in the numpy arrays
    for i, ex in enumerate(example_list):
        enc_batch[i, :] = ex.enc_input[:]
        enc_lens[i] = ex.enc_len
        for j in range(ex.enc_len):
            enc_padding_mask[i][j] = 1

    # For pointer-generator mode, need to store some extra info
    if config.pointer_gen:
        # Determine the max number of in-article OOVs in this batch
        max_art_oovs = max([len(ex.article_oovs) for ex in example_list])
        # Store the in-article OOVs themselves
        art_oovs = [ex.article_oovs for ex in example_list]
        # Store the version of the enc_batch that uses the article OOV ids
        enc_batch_extend_vocab = np.zeros((beam_width, max_enc_seq_len), dtype=np.int32)
        for i, ex in enumerate(example_list):
            enc_batch_extend_vocab[i, :] = ex.enc_input_extend_vocab[:]


    # -------------------------------------------------------- #
    # ---------------- get_input_from_batch  ----------------- #
    # -------------------------------------------------------- #
    # get_input_from_batch
    enc_batch = Variable(torch.from_numpy(enc_batch).long())
    enc_padding_mask = Variable(torch.from_numpy(enc_padding_mask)).float()

    if config.pointer_gen:
        enc_batch_extend_vocab = Variable(torch.from_numpy(enc_batch_extend_vocab).long())
        # max_art_oovs is the max over all the article oov list in the batch
        if max_art_oovs > 0:
            extra_zeros = Variable(torch.zeros((beam_width, max_art_oovs)))

    c_t_1 = Variable(torch.zeros((beam_width, 2 * config.hidden_dim)))

    coverage = None
    if config.is_coverage:
        coverage = Variable(torch.zeros(enc_batch.size()))

    if use_cuda:
        enc_batch = enc_batch.cuda()
        enc_padding_mask = enc_padding_mask.cuda()

        if enc_batch_extend_vocab is not None:
            enc_batch_extend_vocab = enc_batch_extend_vocab.cuda()
        if extra_zeros is not None:
            extra_zeros = extra_zeros.cuda()
        c_t_1 = c_t_1.cuda()

        if coverage is not None:
            coverage = coverage.cuda()

    return (enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_1, coverage), art_oovs

def get_dec_inp_targ_seqs(sequence, max_len, start_id, stop_id):
    inp = [start_id] + sequence[:]
    target = sequence[:]
    if len(inp) > max_len: # truncate
        inp = inp[:max_len]
        target = target[:max_len] # no end_token
    else: # no truncation
        target.append(stop_id) # end token
    assert len(inp) == len(target)
    return inp, target


def main():
    ami_data = load_ami_data('test')
    get_a_batch(ami_data, 0, config.batch_size,
                config.max_enc_steps, config.max_dec_steps, 'short', use_cuda=True)
# main()
