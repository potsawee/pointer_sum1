from __future__ import unicode_literals, print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from data_util import config
from numpy import random

import numpy as np

use_cuda = config.use_gpu and torch.cuda.is_available()

random.seed(123)
torch.manual_seed(123)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(123)

def init_lstm_wt(lstm):
    for names in lstm._all_weights:
        for name in names:
            if name.startswith('weight_'):
                wt = getattr(lstm, name)
                wt.data.uniform_(-config.rand_unif_init_mag, config.rand_unif_init_mag)
            elif name.startswith('bias_'):
                # set forget bias to 1
                bias = getattr(lstm, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data.fill_(0.)
                bias.data[start:end].fill_(1.)

def init_linear_wt(linear):
    linear.weight.data.normal_(std=config.trunc_norm_init_std)
    if linear.bias is not None:
        linear.bias.data.normal_(std=config.trunc_norm_init_std)

def init_wt_normal(wt):
    wt.data.normal_(std=config.trunc_norm_init_std)

def init_wt_unif(wt):
    wt.data.uniform_(-config.rand_unif_init_mag, config.rand_unif_init_mag)

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.emb_dim)
        init_wt_normal(self.embedding.weight)

        self.lstm = nn.LSTM(config.emb_dim, config.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        init_lstm_wt(self.lstm)

        self.W_h = nn.Linear(config.hidden_dim * 2, config.hidden_dim * 2, bias=False)

        if config.is_hierarchical:
            self.sent_lstm = nn.LSTM(config.hidden_dim * 2, config.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
            self.sent_W_h = nn.Linear(config.hidden_dim * 2, config.hidden_dim * 2, bias=False)

    #seq_lens should be in descending order
    def forward(self, input, seq_lens, sent_pos=None):
        embedded = self.embedding(input)

        packed = pack_padded_sequence(embedded, seq_lens, batch_first=True)
        output, hidden = self.lstm(packed)

        encoder_outputs, _ = pad_packed_sequence(output, batch_first=True)  # h dim = B x t_k x n
        encoder_outputs = encoder_outputs.contiguous()

        encoder_feature = encoder_outputs.view(-1, 2*config.hidden_dim)  # B * t_k x 2*hidden_dim
        encoder_feature = self.W_h(encoder_feature)

        if config.is_hierarchical:
            sent_input, sent_lens, sent_mask = self.get_sent_level_input(encoder_outputs, sent_pos)
            sent_packed = pack_padded_sequence(sent_input, sent_lens, batch_first=True, enforce_sorted=False)
            sent_output, sent_hidden = self.sent_lstm(sent_packed)

            sent_enc_outputs, _ = pad_packed_sequence(sent_output, batch_first=True)
            sent_enc_outputs = sent_enc_outputs.contiguous()

            sent_enc_feature = sent_enc_outputs.view(-1, 2*config.hidden_dim)
            sent_enc_feature = self.sent_W_h(sent_enc_feature)

            return encoder_outputs, encoder_feature, hidden, sent_enc_outputs, sent_enc_feature, sent_hidden, sent_mask

        else:
            return encoder_outputs, encoder_feature, hidden

    def get_sent_level_input(self, encoder_outputs, sent_pos):
        batch_size, seq_len, _ = encoder_outputs.size()
        sent_lens = [len(x) for x in sent_pos]
        max_num_sent = max(sent_lens)
        sent_lens = np.array(sent_lens, dtype=np.int32)
        sent_input = torch.zeros((batch_size, max_num_sent, config.hidden_dim*2))

        if use_cuda: sent_input = sent_input.cuda()

        output = encoder_outputs.view(batch_size, seq_len, 2, config.hidden_dim)
        for bi in range(batch_size):
            fw_output = output[bi,:,0,:]
            bw_output = output[bi,:,1,:]

            end_ids = sent_pos[bi]
            start_ids = [0] + [x for x in end_ids[:-1]]

            num_sent = len(end_ids)
            for i in range(num_sent):

                id1 = end_ids[i]
                id2 = start_ids[i]
                sent_input[bi,i,:config.hidden_dim] = fw_output[id1,:]
                sent_input[bi,i,config.hidden_dim:] = bw_output[id2,:]
        sent_mask = [[1.0]*l + [0.0]*(max_num_sent-l) for l in sent_lens]
        sent_mask = torch.tensor(sent_mask)
        if use_cuda: sent_mask = sent_mask.cuda()

        return sent_input, sent_lens, sent_mask

class ReduceState(nn.Module):
    def __init__(self):
        super(ReduceState, self).__init__()

        self.reduce_h = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        init_linear_wt(self.reduce_h)
        self.reduce_c = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        init_linear_wt(self.reduce_c)

        if config.is_hierarchical:
            self.sent_reduce_h = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
            init_linear_wt(self.sent_reduce_h)
            self.sent_reduce_c = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
            init_linear_wt(self.sent_reduce_c)

    def forward1(self, hidden):
        h, c = hidden # h, c dim = num_layers*num_dir x batch_size x hidden_dim
        h_in = h.transpose(0, 1).contiguous().view(-1, config.hidden_dim * 2)
        hidden_reduced_h = F.relu(self.reduce_h(h_in))
        c_in = c.transpose(0, 1).contiguous().view(-1, config.hidden_dim * 2)
        hidden_reduced_c = F.relu(self.reduce_c(c_in))

        return (hidden_reduced_h.unsqueeze(0), hidden_reduced_c.unsqueeze(0)) # h, c dim = 1 x b x hidden_dim

    def forward(self, hidden, sent_hidden):
        h, c = hidden # h, c dim = num_layers*num_dir x batch_size x hidden_dim
        h_in = h.transpose(0, 1).contiguous().view(-1, config.hidden_dim * 2)
        hidden_reduced_h = F.relu(self.reduce_h(h_in))
        c_in = c.transpose(0, 1).contiguous().view(-1, config.hidden_dim * 2)
        hidden_reduced_c = F.relu(self.reduce_c(c_in))

        h2, c2 = sent_hidden
        h2_in = h2.transpose(0, 1).contiguous().view(-1, config.hidden_dim * 2)
        sent_hidden_reduced_h = F.relu(self.sent_reduce_h(h2_in))
        c2_in = c2.transpose(0, 1).contiguous().view(-1, config.hidden_dim * 2)
        sent_hidden_reduced_c = F.relu(self.sent_reduce_c(c2_in))

        return (hidden_reduced_h.unsqueeze(0), hidden_reduced_c.unsqueeze(0)), (sent_hidden_reduced_h.unsqueeze(0), sent_hidden_reduced_c.unsqueeze(0))


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()

        # attention
        if config.is_coverage:
            self.W_c = nn.Linear(1, config.hidden_dim * 2, bias=False)
        self.decode_proj = nn.Linear(config.hidden_dim * 2, config.hidden_dim * 2)
        self.v = nn.Linear(config.hidden_dim * 2, 1, bias=False)

        if config.is_hierarchical:
            # Hierarchical Structure Enabled
            # Sentence-level Attention Mechanism #
            self.sent_decode_proj = nn.Linear(config.hidden_dim * 2, config.hidden_dim * 2)
            self.sent_v = nn.Linear(config.hidden_dim * 2, 1, bias=False)


    def forward1(self, s_t_hat, encoder_outputs, encoder_feature, enc_padding_mask, coverage):
        b, t_k, n = list(encoder_outputs.size())
        dec_fea = self.decode_proj(s_t_hat) # B x 2*hidden_dim
        dec_fea_expanded = dec_fea.unsqueeze(1).expand(b, t_k, n).contiguous() # B x t_k x 2*hidden_dim
        dec_fea_expanded = dec_fea_expanded.view(-1, n)  # B * t_k x 2*hidden_dim

        att_features = encoder_feature + dec_fea_expanded # B * t_k x 2*hidden_dim
        if config.is_coverage:
            coverage_input = coverage.view(-1, 1)  # B * t_k x 1
            coverage_feature = self.W_c(coverage_input)  # B * t_k x 2*hidden_dim
            att_features = att_features + coverage_feature

        e = torch.tanh(att_features) # B * t_k x 2*hidden_dim
        scores = self.v(e)  # B * t_k x 1
        scores = scores.view(-1, t_k)  # B x t_k

        attn_dist_ = F.softmax(scores, dim=1)*enc_padding_mask # B x t_k
        normalization_factor = attn_dist_.sum(1, keepdim=True)
        attn_dist = attn_dist_ / normalization_factor

        attn_dist = attn_dist.unsqueeze(1)  # B x 1 x t_k
        c_t = torch.bmm(attn_dist, encoder_outputs)  # B x 1 x n
        c_t = c_t.view(-1, config.hidden_dim * 2)  # B x 2*hidden_dim

        attn_dist = attn_dist.view(-1, t_k)  # B x t_k

        if config.is_coverage:
            coverage = coverage.view(-1, t_k)
            coverage = coverage + attn_dist

        return c_t, attn_dist, coverage

    def forward(self, s_t_hat, enc_sent_pos, encoder_outputs, encoder_feature, enc_padding_mask,
                sent_s_t_hat, sent_enc_outputs, sent_enc_feature, sent_enc_padding_mask,
                coverage):
        """
        If there is only one LSTM cell in the decoder, sent_s_t_hat = s_t_hat
        """

        b, t_k, n = list(encoder_outputs.size())
        dec_fea = self.decode_proj(s_t_hat) # B x 2*hidden_dim
        dec_fea_expanded = dec_fea.unsqueeze(1).expand(b, t_k, n).contiguous() # B x t_k x 2*hidden_dim
        dec_fea_expanded = dec_fea_expanded.view(-1, n)  # B * t_k x 2*hidden_dim

        att_features = encoder_feature + dec_fea_expanded # B * t_k x 2*hidden_dim
        if config.is_coverage:
            coverage_input = coverage.view(-1, 1)  # B * t_k x 1
            coverage_feature = self.W_c(coverage_input)  # B * t_k x 2*hidden_dim
            att_features = att_features + coverage_feature

        e = torch.tanh(att_features) # B * t_k x 2*hidden_dim
        scores = self.v(e)  # B * t_k x 1
        scores = scores.view(-1, t_k)  # B x t_k

        attn_dist_ = F.softmax(scores, dim=1)*enc_padding_mask # B x t_k
        normalization_factor = attn_dist_.sum(1, keepdim=True)
        attn_dist = attn_dist_ / normalization_factor

        attn_dist = attn_dist.unsqueeze(1)  # B x 1 x t_k
        c_t = torch.bmm(attn_dist, encoder_outputs)  # B x 1 x n
        c_t = c_t.view(-1, config.hidden_dim * 2)  # B x 2*hidden_dim

        attn_dist = attn_dist.view(-1, t_k)  # B x t_k

        # if config.is_hierarchical:
        b, sent_t_k, sent_n = list(sent_enc_outputs.size())
        sent_dec_fea = self.sent_decode_proj(sent_s_t_hat)
        sent_dec_fea_expanded = sent_dec_fea.unsqueeze(1).expand(b, sent_t_k, sent_n).contiguous()
        sent_dec_fea_expanded = sent_dec_fea_expanded.view(-1, n)

        sent_att_features = sent_enc_feature + sent_dec_fea_expanded

        sent_e = torch.tanh(sent_att_features)
        sent_scores = self.sent_v(sent_e)
        sent_scores = sent_scores.view(-1, sent_t_k)

        sent_attn_dist_ = F.softmax(sent_scores, dim=1)*sent_enc_padding_mask
        sent_norm_factor = sent_attn_dist_.sum(1, keepdim=True)
        sent_attn_dist = sent_attn_dist_ / sent_norm_factor

        sent_attn_dist = sent_attn_dist.unsqueeze(1)
        sent_c_t = torch.bmm(sent_attn_dist, sent_enc_outputs)
        sent_c_t = sent_c_t.view(-1, config.hidden_dim * 2)

        sent_attn_dist = sent_attn_dist.view(-1, sent_t_k)

        # Combine the sentence-level & word-level attention scores
        attn_dist_sw = torch.zeros(attn_dist.shape)
        if use_cuda: attn_dist_sw = attn_dist_sw.cuda()

        for bi in range(b):
            end_ids = enc_sent_pos[bi]
            start_ids = [0] + [x for x in end_ids[:-1]]
            for ix in range(len(end_ids)):
                i1 = start_ids[ix]
                i2 = end_ids[ix]
                norm_factor = attn_dist[bi, i1:i2].sum()
                attn_dist_sw[bi, i1:i2] = (attn_dist[bi, i1:i2]/norm_factor) * sent_attn_dist[bi,ix].unsqueeze(-1)

        if config.is_coverage:
            coverage = coverage.view(-1, t_k)
            # coverage = coverage + attn_dist
            coverage = coverage + attn_dist_sw

        return c_t, attn_dist, coverage, sent_c_t, sent_attn_dist


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.attention_network = Attention()

        # decoder
        self.embedding = nn.Embedding(config.vocab_size, config.emb_dim)
        init_wt_normal(self.embedding.weight)

        self.x_context = nn.Linear(config.hidden_dim * 2 + config.emb_dim, config.emb_dim)

        self.lstm = nn.LSTM(config.emb_dim, config.hidden_dim, num_layers=1, batch_first=True, bidirectional=False)
        init_lstm_wt(self.lstm)

        if config.pointer_gen:
            self.p_gen_linear = nn.Linear(config.hidden_dim * 4 + config.emb_dim, 1)

        #p_vocab
        self.out1 = nn.Linear(config.hidden_dim * 3, config.hidden_dim)
        self.out2 = nn.Linear(config.hidden_dim, config.vocab_size)
        init_linear_wt(self.out2)

    def forward1(self, y_t_1, s_t_1, encoder_outputs, encoder_feature, enc_padding_mask,
                c_t_1, extra_zeros, enc_batch_extend_vocab, coverage, step):

        """
        Input:
            y_t_1 = token_id size [batch]
            s_t_1 = (h, c) --- each size [1, batch, hidden_dim]
            encoder_outputs = size [batch, max_enc_steps, 2*hidden]
            encoder_feature = W_h * encoder_outputs, size [batch*max_enc_steps, 2*hidden]
            enc_padding_mask = [batch, max_enc_steps]
            c_t_1 = context_vector size [batch_size, 2*hidden_dim]
            extra_zeros = ???
            enc_batch_extend_vocab = [batch, max_enc_steps]
            coverage = [batch, max_enc_steps]
            step = int starting from 0

        Note that:
            the motivation of x_context is not clear, and not mentioned in See et al 2017
            see this discussion https://github.com/atulkum/pointer_summarizer/issues/1
        """
        if not self.training and step == 0:
            h_decoder, c_decoder = s_t_1
            s_t_hat = torch.cat((h_decoder.view(-1, config.hidden_dim),
                                 c_decoder.view(-1, config.hidden_dim)), 1)  # B x 2*hidden_dim
            c_t, _, coverage_next = self.attention_network.forward1(s_t_hat, encoder_outputs, encoder_feature,
                                                              enc_padding_mask, coverage)
            coverage = coverage_next

        y_t_1_embd = self.embedding(y_t_1)
        x = self.x_context(torch.cat((c_t_1, y_t_1_embd), 1))
        lstm_out, s_t = self.lstm(x.unsqueeze(1), s_t_1)

        h_decoder, c_decoder = s_t
        s_t_hat = torch.cat((h_decoder.view(-1, config.hidden_dim),
                             c_decoder.view(-1, config.hidden_dim)), 1)  # B x 2*hidden_dim
        c_t, attn_dist, coverage_next = self.attention_network.forward1(s_t_hat, encoder_outputs, encoder_feature,
                                                          enc_padding_mask, coverage)

        if self.training or step > 0:
            coverage = coverage_next

        p_gen = None
        if config.pointer_gen:
            p_gen_input = torch.cat((c_t, s_t_hat, x), 1)  # B x (2*2*hidden_dim + emb_dim)
            p_gen = self.p_gen_linear(p_gen_input)
            p_gen = torch.sigmoid(p_gen)

        output = torch.cat((lstm_out.view(-1, config.hidden_dim), c_t), 1) # B x hidden_dim * 3
        output = self.out1(output) # B x hidden_dim

        #output = F.relu(output)

        output = self.out2(output) # B x vocab_size
        vocab_dist = F.softmax(output, dim=1)

        if config.pointer_gen:
            vocab_dist_ = p_gen * vocab_dist
            attn_dist_ = (1 - p_gen) * attn_dist

            if extra_zeros is not None:
                vocab_dist_ = torch.cat([vocab_dist_, extra_zeros], 1)

            final_dist = vocab_dist_.scatter_add(1, enc_batch_extend_vocab, attn_dist_)
        else:
            final_dist = vocab_dist

        return final_dist, s_t, c_t, attn_dist, p_gen, coverage

    def forward(self, y_t_1, s_t_1, enc_sent_pos, encoder_outputs, encoder_feature, enc_padding_mask,
                sent_s_t_1, sent_enc_outputs, sent_enc_feature, sent_enc_padding_mask,
                c_t_1, extra_zeros, enc_batch_extend_vocab, coverage, step):

        """
        note that if the decoder has only one LSTM cell, sent_s_t_1 = s_t_1
        """

        if not self.training and step == 0:
            h_decoder, c_decoder = s_t_1
            s_t_hat = torch.cat((h_decoder.view(-1, config.hidden_dim),
                                 c_decoder.view(-1, config.hidden_dim)), 1)  # B x 2*hidden_dim
            # Hierarchical
            # sent_h_dec, sent_c_dec = sent_s_t_1
            # sent_s_t_hat = torch.cat((sent_h_dec.view(-1, config.hidden_dim),
            #                      sent_c_dec.view(-1, config.hidden_dim)), 1)
            # Currently, there is only one LSTM in the decoder, so sent_s_t_1 is not used!

            c_t, _, coverage_next, sent_c_t, _ = self.attention_network(s_t_hat, enc_sent_pos, encoder_outputs, encoder_feature, enc_padding_mask,
                                                    s_t_hat, sent_enc_outputs, sent_enc_feature, sent_enc_padding_mask, coverage)

            coverage = coverage_next

        y_t_1_embd = self.embedding(y_t_1)
        x = self.x_context(torch.cat((c_t_1, y_t_1_embd), 1))
        lstm_out, s_t = self.lstm(x.unsqueeze(1), s_t_1)

        h_decoder, c_decoder = s_t
        s_t_hat = torch.cat((h_decoder.view(-1, config.hidden_dim),
                             c_decoder.view(-1, config.hidden_dim)), 1)  # B x 2*hidden_dim

        c_t, attn_dist, coverage_next, sent_c_t, sent_attn_dist = self.attention_network(
                                                        s_t_hat, enc_sent_pos, encoder_outputs, encoder_feature, enc_padding_mask,
                                                        s_t_hat, sent_enc_outputs, sent_enc_feature, sent_enc_padding_mask, coverage)

        if self.training or step > 0:
            coverage = coverage_next

        p_gen = None
        if config.pointer_gen:
            p_gen_input = torch.cat((c_t, s_t_hat, x), 1)  # B x (2*2*hidden_dim + emb_dim)
            p_gen = self.p_gen_linear(p_gen_input)
            p_gen = torch.sigmoid(p_gen)

        output = torch.cat((lstm_out.view(-1, config.hidden_dim), c_t), 1) # B x hidden_dim * 3
        output = self.out1(output) # B x hidden_dim

        #output = F.relu(output)

        output = self.out2(output) # B x vocab_size
        vocab_dist = F.softmax(output, dim=1)

        if config.pointer_gen:
            vocab_dist_ = p_gen * vocab_dist
            attn_dist_ = (1 - p_gen) * attn_dist

            if extra_zeros is not None:
                vocab_dist_ = torch.cat([vocab_dist_, extra_zeros], 1)

            final_dist = vocab_dist_.scatter_add(1, enc_batch_extend_vocab, attn_dist_)
        else:
            final_dist = vocab_dist

        return final_dist, s_t, c_t, attn_dist, p_gen, coverage


class Model(object):
    def __init__(self, model_file_path=None, is_eval=False):
        encoder = Encoder()
        decoder = Decoder()
        reduce_state = ReduceState()

        # shared the embedding between encoder and decoder
        decoder.embedding.weight = encoder.embedding.weight
        if is_eval:
            encoder = encoder.eval()
            decoder = decoder.eval()
            reduce_state = reduce_state.eval()

        if use_cuda:
            encoder = encoder.cuda()
            decoder = decoder.cuda()
            reduce_state = reduce_state.cuda()

        self.encoder = encoder
        self.decoder = decoder
        self.reduce_state = reduce_state

        if model_file_path is not None:
            state = torch.load(model_file_path, map_location= lambda storage, location: storage)
            self.encoder.load_state_dict(state['encoder_state_dict'], strict=False)
            self.decoder.load_state_dict(state['decoder_state_dict'], strict=False)
            self.reduce_state.load_state_dict(state['reduce_state_dict'], strict=False)
