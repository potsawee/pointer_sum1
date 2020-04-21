from __future__ import unicode_literals, print_function, division

import os
import sys
import time
import argparse
import pickle
import random
from datetime import datetime

import tensorflow as tf
import torch
from model import Model
from torch.nn.utils import clip_grad_norm_

from torch.optim import Adagrad

from data_util import config
from data_util.batcher import Batcher
from data_util.data import Vocab
from data_util.data import PAD_TOKEN, START_DECODING, STOP_DECODING
from data_util.utils import calc_running_avg_loss
from train_util import get_input_from_batch, get_output_from_batch, get_sent_position

from train_util_ami import load_ami_data, get_a_batch
from train_util_ami import TopicSegment, Utterance

use_cuda = config.use_gpu and torch.cuda.is_available()
random.seed(config.random_seed)

class Train(object):
    def __init__(self):
        if config.is_hierarchical: raise Exception("Hierarchical PGN-AMI not supported!")

        self.vocab = Vocab(config.vocab_path, config.vocab_size)
        self.pad_id   = self.vocab.word2id(PAD_TOKEN)
        self.start_id = self.vocab.word2id(START_DECODING)
        self.stop_id  = self.vocab.word2id(STOP_DECODING)

        self.print_interval = config.print_interval

        train_dir = config.train_dir
        if not os.path.exists(train_dir):
            os.mkdir(train_dir)

        self.model_dir = train_dir
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)

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

    def train_one_batch(self, ami_data, idx):
        # enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_1, coverage = \
        #     get_ami_input_from_batch(batch, use_cuda)
        # dec_batch, dec_padding_mask, max_dec_len, dec_lens_var, target_batch = \
        #     get_ami_output_from_batch(batch, use_cuda)

        enc_pack, dec_pack = get_a_batch(
                    ami_data, idx, self.vocab,
                    config.batch_size, config.max_enc_steps, config.max_dec_steps,
                    self.start_id, self.stop_id, self.pad_id,
                    sum_type='short', use_cuda=use_cuda)
        enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_1, coverage = enc_pack
        dec_batch, dec_padding_mask, max_dec_len, dec_lens_var, target_batch = dec_pack

        self.optimizer.zero_grad()

        encoder_outputs, encoder_feature, encoder_hidden = self.model.encoder(enc_batch, enc_lens)
        s_t_1 = self.model.reduce_state.forward1(encoder_hidden)

        step_losses = []
        for di in range(min(max_dec_len, config.max_dec_steps)):
            y_t_1 = dec_batch[:, di]  # Teacher forcing

            final_dist, s_t_1,  c_t_1, attn_dist, p_gen, next_coverage = self.model.decoder.forward1(y_t_1, s_t_1,
                                                        encoder_outputs, encoder_feature, enc_padding_mask, c_t_1,
                                                        extra_zeros, enc_batch_extend_vocab,
                                                                           coverage, di)


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

        loss.backward()

        clip_grad_norm_(self.model.encoder.parameters(), config.max_grad_norm)
        clip_grad_norm_(self.model.decoder.parameters(), config.max_grad_norm)
        clip_grad_norm_(self.model.reduce_state.parameters(), config.max_grad_norm)

        self.optimizer.step()

        return loss.item()

    def trainIters(self, n_iters, model_file_path=None):
        iter, running_avg_loss = self.setup_train(model_file_path)
        sys.stdout.flush()

        ami_data   = load_ami_data('train')
        valid_data = load_ami_data('valid')
        # make the training data 100
        random.shuffle(valid_data)
        ami_data.extend(valid_data[:6])
        valid_data = valid_data[6:]

        num_batches = len(ami_data)
        idx = 0

        # validation & stopping
        best_valid_loss = 1000000000
        stop_counter    = 0

        while iter < n_iters:
            if idx == 0:
                print("shuffle training data")
                random.shuffle(ami_data)

            loss = self.train_one_batch(ami_data, idx)

            running_avg_loss = calc_running_avg_loss(loss, running_avg_loss, iter)

            iter += 1
            idx  += config.batch_size
            if idx == num_batches: idx = 0

            if iter % self.print_interval == 0:
                print("[{}] iter {}, loss: {:.5f}".format(str(datetime.now()), iter, loss))
                sys.stdout.flush()

            if iter % config.save_every == 0:
                self.save_model(running_avg_loss, iter)

            if iter % config.eval_every == 0:
                valid_loss = self.run_eval(valid_data)
                print("valid_loss = {:.5f}".format(valid_loss))
                if valid_loss < best_valid_loss:
                    stop_counter    = 0
                    best_valid_loss = valid_loss
                    print("VALID better")
                else:
                    stop_counter += 1
                    print("VALID NOT better, counter = {}".format(stop_counter))
                    if stop_counter == config.stop_after:
                        print("Stop training")
                        return

        print("Finished training!")


    def eval_one_batch(self, eval_data, idx):

        enc_pack, dec_pack = get_a_batch(
                    eval_data, idx, self.vocab,
                    1, config.max_enc_steps, config.max_dec_steps,
                    self.start_id, self.stop_id, self.pad_id,
                    sum_type='short', use_cuda=use_cuda)

        enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_1, coverage = enc_pack
        dec_batch, dec_padding_mask, max_dec_len, dec_lens_var, target_batch = dec_pack

        encoder_outputs, encoder_feature, encoder_hidden = self.model.encoder(enc_batch, enc_lens)
        s_t_1 = self.model.reduce_state.forward1(encoder_hidden)

        step_losses = []
        for di in range(min(max_dec_len, config.max_dec_steps)):
            y_t_1 = dec_batch[:, di]  # Teacher forcing
            final_dist, s_t_1, c_t_1,attn_dist, p_gen, next_coverage = self.model.decoder.forward1(y_t_1, s_t_1,
                                                        encoder_outputs, encoder_feature, enc_padding_mask, c_t_1,
                                                        extra_zeros, enc_batch_extend_vocab, coverage, di)

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

    def run_eval(self, eval_data):
        running_avg_loss, iter = 0, 0
        batch_losses = []
        num_batches = len(eval_data)
        print("valid data size = {}".format(num_batches))
        for idx in range(num_batches):
            loss = self.eval_one_batch(eval_data, idx)
            batch_losses.append(loss)
            running_avg_loss = calc_running_avg_loss(loss, running_avg_loss, iter)
            print("#", end="")
            sys.stdout.flush()
        print()

        avg_loss = sum(batch_losses) / len(batch_losses)
        return avg_loss

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
