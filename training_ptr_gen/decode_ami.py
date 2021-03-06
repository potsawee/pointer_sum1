#Except for the pytorch part content of this file is copied from https://github.com/abisee/pointer-generator/blob/master/

from __future__ import unicode_literals, print_function, division

import sys
# reload(sys)
# sys.setdefaultencoding('utf8')

import os
import time
import pickle
import random
from pathlib import Path
from datetime import datetime

import torch
from torch.autograd import Variable

from data_util.batcher import Batcher
from data_util.data import Vocab
from data_util.data import PAD_TOKEN, START_DECODING, STOP_DECODING
from data_util import data, config
from model import Model
from data_util.utils import write_for_rouge, rouge_eval, rouge_log
from train_util import get_input_from_batch, get_sent_position

from train_util_ami import load_ami_data, get_a_batch_decode
from train_util_ami import TopicSegment, Utterance

use_cuda = config.use_gpu and torch.cuda.is_available()

class Beam(object):
  def __init__(self, tokens, log_probs, state, context, coverage):
    self.tokens = tokens
    self.log_probs = log_probs
    self.state = state
    self.context = context
    self.coverage = coverage

  def extend(self, token, log_prob, state, context, coverage):
    return Beam(tokens = self.tokens + [token],
                      log_probs = self.log_probs + [log_prob],
                      state = state,
                      context = context,
                      coverage = coverage)

  @property
  def latest_token(self):
    return self.tokens[-1]

  @property
  def avg_log_prob(self):
    return sum(self.log_probs) / len(self.tokens)


class BeamSearch(object):
    def __init__(self, model_file_path):
        model_name = os.path.basename(model_file_path)
        self._decode_dir = os.path.join(config.decode_dir, model_name)
        self._decode_dir = os.path.splitext(self._decode_dir)[0]
        self._rouge_ref_dir = os.path.join(self._decode_dir, 'rouge_ref')
        self._rouge_dec_dir = os.path.join(self._decode_dir, 'rouge_dec_dir')

        for p in [self._decode_dir, self._rouge_ref_dir, self._rouge_dec_dir]:
            Path(p).mkdir(parents=True, exist_ok=True)

        self.vocab = Vocab(config.vocab_path, config.vocab_size)
        self.pad_id   = self.vocab.word2id(PAD_TOKEN)
        self.start_id = self.vocab.word2id(START_DECODING)
        self.stop_id  = self.vocab.word2id(STOP_DECODING)

        self.model = Model(model_file_path, is_eval=True)

    def sort_beams(self, beams):
        return sorted(beams, key=lambda h: h.avg_log_prob, reverse=True)

    def if_already_exists(self, idx):
        decoded_file = os.path.join(self._rouge_dec_dir, "file.{}.txt".format(idx))
        return os.path.isfile(decoded_file)

    def decode(self, file_id_start, file_id_stop, ami_id='191209'):
        print("AMI transcription:", ami_id)

        test_data = load_ami_data(ami_id, 'test')

        # do this for faster stack CPU machines - to replace those that fail!!
        idx_list = [i for i in range(file_id_start, file_id_stop)]
        random.shuffle(idx_list)
        for idx in idx_list:

        # for idx in range(file_id_start, file_id_stop):
            # check if this is written already
            if self.if_already_exists(idx):
                print("ID {} already exists".format(idx))
                continue

            # Run beam search to get best Hypothesis
            best_summary, art_oovs = self.beam_search(test_data, idx)

            # Extract the output ids from the hypothesis and convert back to words
            output_ids = [int(t) for t in best_summary.tokens[1:]]
            decoded_words = data.outputids2words(output_ids, self.vocab,
                                                 (art_oovs[0] if config.pointer_gen else None))

            # Remove the [STOP] token from decoded_words, if necessary
            try:
                fst_stop_idx = decoded_words.index(data.STOP_DECODING)
                decoded_words = decoded_words[:fst_stop_idx]
            except ValueError:
                decoded_words = decoded_words

            # original_abstract_sents = batch.original_abstracts_sents[0]
            original_abstract_sents = []

            write_for_rouge(original_abstract_sents, decoded_words, idx,
                            self._rouge_ref_dir, self._rouge_dec_dir)

            print("decoded idx = {}".format(idx))
        print("Finished decoding idx [{},{})".format(file_id_start, file_id_stop))

        # print("Starting ROUGE eval...")
        # results_dict = rouge_eval(self._rouge_ref_dir, self._rouge_dec_dir)
        # rouge_log(results_dict, self._decode_dir)


    def beam_search(self, test_data, idx):
        # batch should have only one example
        # enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_0, coverage_t_0 = \
        #       get_input_from_batch(batch, use_cuda)

        enc_pack, art_oovs = get_a_batch_decode(
                test_data, idx, self.vocab,
                config.beam_size, config.max_enc_steps, config.max_dec_steps,
                self.start_id, self.stop_id, self.pad_id,
                sum_type='short', use_cuda=use_cuda)
        enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_0, coverage_t_0 = enc_pack
        encoder_outputs, encoder_feature, encoder_hidden = self.model.encoder(enc_batch, enc_lens)
        s_t_0 = self.model.reduce_state.forward1(encoder_hidden)

        dec_h, dec_c = s_t_0 # 1 x 2*hidden_size
        dec_h = dec_h.squeeze()
        dec_c = dec_c.squeeze()

        # decoder batch preparation, it has beam_size example initially everything is repeated
        beams = [Beam(tokens=[self.vocab.word2id(data.START_DECODING)],
                      log_probs=[0.0],
                      state=(dec_h[0], dec_c[0]),
                      context = c_t_0[0],
                      coverage=(coverage_t_0[0] if config.is_coverage else None))
                 for _ in range(config.beam_size)]

        results = []
        steps = 0
        while steps < config.max_dec_steps and len(results) < config.beam_size:
            latest_tokens = [h.latest_token for h in beams]
            latest_tokens = [t if t < self.vocab.size() else self.vocab.word2id(data.UNKNOWN_TOKEN) \
                             for t in latest_tokens]
            y_t_1 = Variable(torch.LongTensor(latest_tokens))
            if use_cuda: y_t_1 = y_t_1.cuda()
            all_state_h =[]
            all_state_c = []

            all_context = []

            for h in beams:
                state_h, state_c = h.state
                all_state_h.append(state_h)
                all_state_c.append(state_c)

                all_context.append(h.context)

            s_t_1 = (torch.stack(all_state_h, 0).unsqueeze(0), torch.stack(all_state_c, 0).unsqueeze(0))
            c_t_1 = torch.stack(all_context, 0)

            coverage_t_1 = None
            if config.is_coverage:
                all_coverage = []
                for h in beams:
                    all_coverage.append(h.coverage)
                coverage_t_1 = torch.stack(all_coverage, 0)

            final_dist, s_t, c_t, attn_dist, p_gen, coverage_t = self.model.decoder.forward1(y_t_1, s_t_1,
                                                        encoder_outputs, encoder_feature, enc_padding_mask, c_t_1,
                                                        extra_zeros, enc_batch_extend_vocab, coverage_t_1, steps)

            log_probs = torch.log(final_dist)
            topk_log_probs, topk_ids = torch.topk(log_probs, config.beam_size * 2)

            dec_h, dec_c = s_t
            dec_h = dec_h.squeeze()
            dec_c = dec_c.squeeze()

            all_beams = []
            num_orig_beams = 1 if steps == 0 else len(beams)
            for i in range(num_orig_beams):
                h = beams[i]
                state_i = (dec_h[i], dec_c[i])
                context_i = c_t[i]
                coverage_i = (coverage_t[i] if config.is_coverage else None)

                for j in range(config.beam_size * 2):  # for each of the top 2*beam_size hyps:
                    new_beam = h.extend(token=topk_ids[i, j].item(),
                                   log_prob=topk_log_probs[i, j].item(),
                                   state=state_i,
                                   context=context_i,
                                   coverage=coverage_i)
                    all_beams.append(new_beam)

            beams = []
            for h in self.sort_beams(all_beams):
                if h.latest_token == self.vocab.word2id(data.STOP_DECODING):
                    if steps >= config.min_dec_steps:
                        results.append(h)
                else:
                    beams.append(h)
                if len(beams) == config.beam_size or len(results) == config.beam_size:
                    break

            steps += 1

        if len(results) == 0:
            results = beams

        beams_sorted = self.sort_beams(results)

        return beams_sorted[0], art_oovs

if __name__ == '__main__':
    model_filename = sys.argv[1]
    file_id_start = int(sys.argv[2])
    if len(sys.argv) == 4:
        file_id_stop = int(sys.argv[3])
    else:
        file_id_stop = file_id_start+100

    beam_Search_processor = BeamSearch(model_filename)
    beam_Search_processor.decode(file_id_start, file_id_stop, ami_id='asr-200124')
    """
    ami_id:
        191209     = manual transcription
        asr-200214 = Linlin's ASR output (CUED internal, WER = 20%)
        asr-200124 = AMI ASR Official Release
    """
