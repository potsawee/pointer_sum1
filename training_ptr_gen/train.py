from __future__ import unicode_literals, print_function, division

import os
import sys
import time
import argparse
from datetime import datetime

import tensorflow as tf
import torch
from model import Model
from torch.nn.utils import clip_grad_norm_

from torch.optim import Adagrad

from data_util import config
from data_util.batcher import Batcher
from data_util.data import Vocab
from data_util.utils import calc_running_avg_loss
from train_util import get_input_from_batch, get_output_from_batch, get_sent_position

use_cuda = config.use_gpu and torch.cuda.is_available()

class Train(object):
    def __init__(self):
        self.vocab = Vocab(config.vocab_path, config.vocab_size)

        self.batcher = Batcher(config.train_data_path, self.vocab, mode='train',
                               batch_size=config.batch_size, single_pass=False)
        # print("MODE MUST BE train")
        # time.sleep(15)
        self.print_interval = config.print_interval

        train_dir = config.train_dir
        if not os.path.exists(train_dir):
            os.mkdir(train_dir)

        self.model_dir = train_dir
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)

        # self.summary_writer = tf.compat.v1.summary.FileWriter(train_dir)

    def save_model(self, running_avg_loss, iter):
        state = {
            'iter': iter,
            'encoder_state_dict': self.model.encoder.state_dict(),
            'decoder_state_dict': self.model.decoder.state_dict(),
            'reduce_state_dict': self.model.reduce_state.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'current_loss': running_avg_loss
        }
        model_save_path = os.path.join(self.model_dir, 'iter{}.pt'.format(iter))
        torch.save(state, model_save_path)

    def setup_train(self, model_file_path=None):
        self.model = Model(model_file_path)

        params = list(self.model.encoder.parameters()) + list(self.model.decoder.parameters()) + \
                 list(self.model.reduce_state.parameters())
        initial_lr = config.lr_coverage if config.is_coverage else config.lr
        self.optimizer = Adagrad(params, lr=initial_lr, initial_accumulator_value=config.adagrad_init_acc)

        start_iter, start_loss = 0, 0

        if model_file_path is not None:
            state = torch.load(model_file_path, map_location= lambda storage, location: storage)
            start_iter = state['iter']
            start_loss = state['current_loss']

            if not config.is_coverage:
                self.optimizer.load_state_dict(state['optimizer'])
                if use_cuda:
                    for state in self.optimizer.state.values():
                        for k, v in state.items():
                            if torch.is_tensor(v):
                                state[k] = v.cuda()

        return start_iter, start_loss

    def train_one_batch(self, batch):
        enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_1, coverage = \
            get_input_from_batch(batch, use_cuda)
        dec_batch, dec_padding_mask, max_dec_len, dec_lens_var, target_batch = \
            get_output_from_batch(batch, use_cuda)

        self.optimizer.zero_grad()

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
                # start = datetime.now()

                final_dist, s_t_1,  c_t_1, attn_dist, p_gen, next_coverage = self.model.decoder.forward1(y_t_1, s_t_1,
                                                            encoder_outputs, encoder_feature, enc_padding_mask, c_t_1,
                                                            extra_zeros, enc_batch_extend_vocab,
                                                                               coverage, di)
                # print('NO HIER Time: ',datetime.now() - start)
                # import pdb; pdb.set_trace()
            else:
                # start = datetime.now()

                final_dist, s_t_1,  c_t_1, attn_dist, p_gen, next_coverage = self.model.decoder(y_t_1, s_t_1, enc_sent_pos,
                                                            encoder_outputs, encoder_feature, enc_padding_mask,
                                                            sent_s_t_1, sent_enc_outputs, sent_enc_feature, sent_enc_padding_mask,
                                                            c_t_1, extra_zeros, enc_batch_extend_vocab, coverage, di)
                # print('DO HIER Time: ',datetime.now() - start)
                # import pdb; pdb.set_trace()


            target = target_batch[:, di]
            gold_probs = torch.gather(final_dist, 1, target.unsqueeze(1)).squeeze()
            step_loss = -torch.log(gold_probs + config.eps)
            if config.is_coverage:
                step_coverage_loss = torch.sum(torch.min(attn_dist, coverage), 1)
                step_loss = step_loss + config.cov_loss_wt * step_coverage_loss
                coverage = next_coverage

            step_mask = dec_padding_mask[:, di]
            step_loss = step_loss * step_mask
            step_losses.append(step_loss)

        sum_losses = torch.sum(torch.stack(step_losses, 1), 1)
        batch_avg_loss = sum_losses/dec_lens_var
        loss = torch.mean(batch_avg_loss)

        # start = datatime.now()
        loss.backward()
        # print('{} HIER Time: {}'.format(config.is_hierarchical ,datetime.now() - start))
        # import pdb; pdb.set_trace()

        self.norm = clip_grad_norm_(self.model.encoder.parameters(), config.max_grad_norm)
        clip_grad_norm_(self.model.decoder.parameters(), config.max_grad_norm)
        clip_grad_norm_(self.model.reduce_state.parameters(), config.max_grad_norm)

        self.optimizer.step()

        return loss.item()

    def trainIters(self, n_iters, model_file_path=None):
        iter, running_avg_loss = self.setup_train(model_file_path)
        sys.stdout.flush()
        while iter < n_iters:
            batch = self.batcher.next_batch()
            loss = self.train_one_batch(batch)

            # running_avg_loss = calc_running_avg_loss(loss, running_avg_loss, self.summary_writer, iter)
            running_avg_loss = calc_running_avg_loss(loss, running_avg_loss, iter)

            iter += 1

            # if iter % 100 == 0:
            #     self.summary_writer.flush()

            if iter % self.print_interval == 0:
                print("[{}] iter {}, loss: {:.5f}".format(str(datetime.now()), iter, loss))
                sys.stdout.flush()

            if iter % config.save_every == 0:
                self.save_model(running_avg_loss, iter)

        print("Finished training!")

if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description="Train script")
    # parser.add_argument("-m",
    #                     dest="model_file_path",
    #                     required=False,
    #                     default=None,
    #                     help="Model file for retraining (default: None).")
    #
    # args = parser.parse_args()
    # train_processor = Train()
    # train_processor.trainIters(config.max_iterations, args.model_file_path)

    train_processor = Train()
    train_processor.trainIters(config.max_iterations, config.model_file_path)
