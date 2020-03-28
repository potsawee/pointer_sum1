from __future__ import unicode_literals, print_function, division

import os
import time
import sys
from datetime import datetime

import tensorflow as tf
import torch

from data_util import config
from data_util.batcher import Batcher
from data_util.data import Vocab

from data_util.utils import calc_running_avg_loss
from train_util import get_input_from_batch, get_output_from_batch, get_sent_position
from model import Model

use_cuda = config.use_gpu and torch.cuda.is_available()

class Evaluate(object):
    def __init__(self, model_file_path):
        self.vocab = Vocab(config.vocab_path, config.vocab_size)
        self.batcher = Batcher(config.eval_data_path, self.vocab, mode='eval',
                               batch_size=config.batch_size, single_pass=True)
        # time.sleep(15)
        model_name = os.path.basename(model_file_path)
        # eval_dir = os.path.join(config.log_root, 'eval_%s' % (model_name))
        # if not os.path.exists(eval_dir):
        #     os.mkdir(eval_dir)
        # self.summary_writer = tf.summary.FileWriter(eval_dir)
        self.model_file_path = model_file_path
        self.model = Model(model_file_path, is_eval=True)

    def eval_one_batch(self, batch):
        enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_1, coverage = \
            get_input_from_batch(batch, use_cuda)
        dec_batch, dec_padding_mask, max_dec_len, dec_lens_var, target_batch = \
            get_output_from_batch(batch, use_cuda)

        if not config.is_hierarchical:
            encoder_outputs, encoder_feature, encoder_hidden = self.model.encoder(enc_batch, enc_lens)
            s_t_1 = self.model.reduce_state.forward1(encoder_hidden)

        else:
            stop_id = self.vocab.word2id('.')
            enc_sent_pos = get_sent_position(enc_batch, stop_id)
            dec_sent_pos = get_sent_position(dec_batch, stop_id)

            encoder_outputs, encoder_feature, encoder_hidden, sent_enc_outputs, sent_enc_feature, sent_enc_hidden, sent_enc_padding_mask = \
                                                                    self.model.encoder(enc_batch, enc_lens, enc_sent_pos)
            s_t_1, sent_s_t_1 = self.model.reduce_state(encoder_hidden, sent_enc_hidden)


        step_losses = []
        for di in range(min(max_dec_len, config.max_dec_steps)):
            y_t_1 = dec_batch[:, di]  # Teacher forcing
            if not config.is_hierarchical:
                final_dist, s_t_1, c_t_1,attn_dist, p_gen, next_coverage = self.model.decoder.forward1(y_t_1, s_t_1,
                                                            encoder_outputs, encoder_feature, enc_padding_mask, c_t_1,
                                                            extra_zeros, enc_batch_extend_vocab, coverage, di)

            else:

                final_dist, s_t_1,  c_t_1, attn_dist, p_gen, next_coverage = self.model.decoder(y_t_1, s_t_1, enc_sent_pos,
                                                            encoder_outputs, encoder_feature, enc_padding_mask,
                                                            sent_s_t_1, sent_enc_outputs, sent_enc_feature, sent_enc_padding_mask,
                                                            c_t_1, extra_zeros, enc_batch_extend_vocab, coverage, di)

            target = target_batch[:, di]
            gold_probs = torch.gather(final_dist, dim=1, index=target.unsqueeze(1)).squeeze()
            step_loss = -torch.log(gold_probs + config.eps)
            if config.is_coverage:
                step_coverage_loss = torch.sum(torch.min(attn_dist, coverage), 1)
                step_loss = step_loss + config.cov_loss_wt * step_coverage_loss
                coverage = next_coverage

            step_mask = dec_padding_mask[:, di]
            step_loss = step_loss * step_mask
            step_losses.append(step_loss)

        sum_step_losses = torch.sum(torch.stack(step_losses, 1), 1)
        batch_avg_loss = sum_step_losses / dec_lens_var
        loss = torch.mean(batch_avg_loss)

        return loss.data.item()

    def run_eval(self):
        running_avg_loss, iter = 0, 0
        batch_losses = []
        # while batch is not None:
        for _ in range(835):
            batch = self.batcher.next_batch()

            loss = self.eval_one_batch(batch)
            batch_losses.append(loss)
            # running_avg_loss = calc_running_avg_loss(loss, running_avg_loss, self.summary_writer, iter)
            running_avg_loss = calc_running_avg_loss(loss, running_avg_loss, iter)
            iter += 1

            # if iter % 100 == 0:
            #     self.summary_writer.flush()

            print_interval = 10
            if iter % print_interval == 0:
                print("[{}] iter {}, loss: {:.5f}".format(str(datetime.now()), iter, loss))

        avg_loss = sum(batch_losses) / len(batch_losses)
        print("Finished Eval for Model {}: Avg Loss = {:.5f}".format(self.model_file_path, avg_loss))

if __name__ == '__main__':
    model_filename = sys.argv[1]
    eval_processor = Evaluate(model_filename)
    eval_processor.run_eval()
