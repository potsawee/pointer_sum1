from data_util.data import Vocab
from data_util import data
from data_util import config
from training_ptr_gen.model import Model

from torch.autograd import Variable
import torch

from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
import numpy as np

LONG_BORING_TENNIS_ARTICLE = """
 Andy Murray  came close to giving himself some extra preparation time for his w
edding next week before ensuring that he still has unfinished tennis business to
 attend to. The world No 4 is into the semi-finals of the Miami Open, but not be
fore getting a scare from 21 year-old Austrian Dominic Thiem, who pushed him to
4-4 in the second set before going down 3-6 6-4, 6-1 in an hour and three quarte
rs. Murray was awaiting the winner from the last eight match between Tomas Berdy
ch and Argentina's Juan Monaco. Prior to this tournament Thiem lost in the secon
d round of a Challenger event to soon-to-be new Brit Aljaz Bedene. Andy Murray p
umps his first after defeating Dominic Thiem to reach the Miami Open semi finals
 . Muray throws his sweatband into the crowd after completing a 3-6, 6-4, 6-1 vi
ctory in Florida . Murray shakes hands with Thiem who he described as a 'strong
guy' after the game . And Murray has a fairly simple message for any of his fell
ow British tennis players who might be agitated about his imminent arrival into
the home ranks: don't complain. Instead the British No 1 believes his colleagues
 should use the assimilation of the world number 83, originally from Slovenia, a
s motivation to better themselves. At present any grumbles are happening in priv
ate, and Bedene's present ineligibility for the Davis Cup team has made it less
of an issue, although that could change if his appeal to play is allowed by the
International Tennis Federation. Murray thinks anyone questioning the move, now
it has become official, would be better working on getting their ranking closer
to his. 'If he was 500 in the world they wouldn't be that fussed about it but ob
viously he threatens their position a bit,' said the 27 year-old Scot. ' and he'
s obviously the British number two, comfortably. 'So they can complain but the b
est thing to do is use it in the right way and accept it for what it is, and try
 to use it as motivation whether they agree with it or not. He's British now so
they've just got to deal with it. Murray stretches for a return after starting h
is quarter final match slowly on the show court . Thiem held nothing back as he
raced through the opening set, winning it 6-3 with a single break . The young Au
strian is considered to be one of the hottest prospects on the ATP Tour . 'I wou
ld hope that all the guys who are below him now like James (Ward) , Kyle (Edmund
) , Liam (Broady) they will use it as motivation. If he becomes eligible for Dav
is Cup then those guys are going to have to prove themselves. 'It can only be se
en as a positive for those guys using it to try to get better. He's a good playe
r but so are James and Kyle and Liam has improved. Aljaz is there, he's on the t
our every week, the other guys aren't quite there yet.' For the first time Murra
y, who has an encyclopaedic knowledge of the top 100, gave his opinion of Bedene
: 'He's a good player with a very good serve. He's a legitimate top 100 player,
when he plays Challengers he's there or thereabouts, when he plays on the main t
our he wins matches, it's not like he turns up and always loses in the first rou
nd. Murray's fiancee was once again watching from the stands shaded by a huge br
immed hat . Kim Sears flashes her enormous diamond engagement ring while watchin
g her beau on court . 'He had a bad injury last year (wrist) but has recovered w
ell. I would imagine he would keep moving up the rankings although I don't know
exactly how high he can go. I've practised with him a couple of times, I haven't
 seen him play loads, but when you serve as well as he does it helps. I would im
agine he' s going to be comfortably in the top 70 or 80 in the world for a while
.' It is understood the Lawn Tennis Association will give background support to
his case regarding the Davis Cup but have made it clear that the onus is on him
to lead the way. An official statement said: 'To have another player in the men'
s top 100 is clearly a positive thing for British tennis and so we very much wel
come Aljaz's change in citizenship.' The last comparable switch came twenty year
s ago when Greg Rusedski arrived from Canada. It was by no means universally pop
ular but, like Bedene, he pledged that he was in for the long haul and, in fairn
ess to him, he proved true to his word. Loising the first set shocked Murray int
o life as he raced to a commanding lead in the second . The No 3 seed sent over
a few glaring looks towards his team before winning the second set . Murray had
to put such matters aside as he tackled the unusually talented Thiem, a delight
to watch. Coached by Boris Becker's veteran mentor Gunter Bresnik, he slightly r
esembles Andy Roddick and hits with similar power but more elegance. His single
handed backhand is a thing of rare beauty. However, he has had a mediocre season
 coming into this event and there was little to forewarn of his glorious shotmak
ing that seemed to catch Murray unawares early on. The world No 4 looked to have
 worked him out in the second, but then suffered one of his periopdic mental lap
ses and let him back in from 4-1 before closing it out with a break. After break
ing him for 3-1 in the decider the Austrian whirlwind burnt itself out. 'He's a
strong guy who hits the ball hard and it became a very physical match,' said Mur
ray. Murray was presented with a celebratory cake after winning his 500th match
in the previous round .
""".replace('\n','')


class ExampleInput(object):
    def __init__(self, enc_input, enc_len, enc_input_extend_vocab, article_oovs):

        self.enc_input = enc_input
        self.enc_len   = enc_len
        self.enc_input_extend_vocab = enc_input_extend_vocab
        self.article_oovs = article_oovs

    def pad_encoder_input(self, max_len, pad_id):
        while len(self.enc_input) < max_len:
            self.enc_input.append(pad_id)
        if config.pointer_gen:
            while len(self.enc_input_extend_vocab) < max_len:
                self.enc_input_extend_vocab.append(pad_id)

class EncoderBatch(object):
    def __init__(self, enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_1, coverage, art_oovs):
        self.enc_batch = enc_batch
        self.enc_padding_mask = enc_padding_mask
        self.enc_lens = enc_lens
        self.enc_batch_extend_vocab = enc_batch_extend_vocab
        self.extra_zeros = extra_zeros
        self.c_t_1 = c_t_1
        self.coverage = coverage
        self.art_oovs = art_oovs

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

class PGNTokenizer(object):
    def __init__(self):
        self.vocab = Vocab(config.vocab_path, config.vocab_size)
        self.pad_id   = self.vocab.word2id(data.PAD_TOKEN)
        # START_ID = vocab.word2id(data.START_DECODING)
        # STOP_ID  = vocab.word2id(data.STOP_DECODING)

    def get_decoding_batches_from_docs(self, docs, beam_size, use_cuda=False):
        batch_size = len(docs)
        example_input_list = [None for _ in range(batch_size)]
        for i in range(batch_size):
            doc = docs[i]
            article_words =  word_tokenize(doc)
            if len(article_words) > config.max_enc_steps:
                article_words = article_words[:config.max_enc_steps]
            enc_len = len(article_words)
            enc_input = [self.vocab.word2id(w) for w in article_words]

            # If using pointer-generator mode, we need to store some extra info
            if config.pointer_gen:
                # Store a version of the enc_input where in-article OOVs are represented by their temporary OOV id; also store the in-article OOVs words themselves
                enc_input_extend_vocab, article_oovs = data.article2ids(article_words, self.vocab)
            else:
                enc_input_extend_vocab = None
                article_oovs = None

            example_input = ExampleInput(enc_input, enc_len, enc_input_extend_vocab, article_oovs)
            example_input_list[i] = example_input

        max_enc_seq_len = max([ex.enc_len for ex in example_input_list])
        for ex in example_input_list:
            ex.pad_encoder_input(max_enc_seq_len, self.pad_id)

        # Initialize the numpy arrays
        # Note: our enc_batch can have different length (second dimension) for each batch because we use dynamic_rnn for the encoder.
        enc_batch = np.zeros((batch_size, max_enc_seq_len), dtype=np.int32)
        enc_lens = np.zeros((batch_size), dtype=np.int32)
        enc_padding_mask = np.zeros((batch_size, max_enc_seq_len), dtype=np.float32)
        # Fill in the numpy arrays
        for i, ex in enumerate(example_input_list):
            enc_batch[i, :] = ex.enc_input[:]
            enc_lens[i] = ex.enc_len
            for j in range(ex.enc_len):
                enc_padding_mask[i][j] = 1

        # For pointer-generator mode, need to store some extra info
        if config.pointer_gen:
            # Determine the max number of in-article OOVs in this batch
            # max_art_oovs = max([len(ex.article_oovs) for ex in example_input_list])

            # Store the in-article OOVs themselves
            # art_oovs = [ex.article_oovs for ex in example_input_list]

            # Store the version of the enc_batch that uses the article OOV ids
            enc_batch_extend_vocab = np.zeros((batch_size, max_enc_seq_len), dtype=np.int32)
            for i, ex in enumerate(example_input_list):
                enc_batch_extend_vocab[i, :] = ex.enc_input_extend_vocab[:]


        # need to duplicate each doc for beam_size times, one batch = [beam_size,...]
        decoding_batches = [None for _ in range(batch_size)]
        for i in range(batch_size):
            tiled_enc_batch_i        = np.tile(enc_batch[i], [beam_size, 1])
            tiled_enc_padding_mask_i = np.tile(enc_padding_mask[i], [beam_size, 1])
            tiled_enc_lens_i = np.tile(enc_lens[i], [beam_size,])

            tiled_enc_batch_i = Variable(torch.from_numpy(tiled_enc_batch_i).long())
            tiled_enc_padding_mask_i = Variable(torch.from_numpy(tiled_enc_padding_mask_i)).float()


            if config.pointer_gen:
                tiled_enc_batch_extend_vocab_i = np.tile(enc_batch_extend_vocab[i], [beam_size, 1])
                tiled_enc_batch_extend_vocab_i = Variable(torch.from_numpy(tiled_enc_batch_extend_vocab_i).long())

                # max_art_oovs is the max over all the article oov list in the batch
                max_art_oovs = len(example_input_list[i].article_oovs)
                if max_art_oovs > 0:
                    extra_zeros = Variable(torch.zeros((beam_size, max_art_oovs)))

                art_oovs = example_input_list[i].article_oovs

            c_t_1 = Variable(torch.zeros((beam_size, 2 * config.hidden_dim)))

            coverage = None
            if config.is_coverage:
                coverage = Variable(torch.zeros(tiled_enc_batch_i.size()))

            if use_cuda:
                tiled_enc_batch_i = tiled_enc_batch_i.cuda()
                tiled_enc_padding_mask_i = tiled_enc_padding_mask_i.cuda()

                if tiled_enc_batch_extend_vocab_i is not None:
                    tiled_enc_batch_extend_vocab_i = tiled_enc_batch_extend_vocab_i.cuda()
                if extra_zeros is not None:
                    extra_zeros = extra_zeros.cuda()
                c_t_1 = c_t_1.cuda()

                if coverage is not None:
                    coverage = coverage.cuda()

            decoding_batch = EncoderBatch(tiled_enc_batch_i, tiled_enc_padding_mask_i, tiled_enc_lens_i,
                                    tiled_enc_batch_extend_vocab_i, extra_zeros, c_t_1, coverage, art_oovs)

            decoding_batches[i] = decoding_batch

        return decoding_batches


class PGNwithCoverage(object):
    def __init__(self, model_name):
        if model_name not in ["PTR_COV2"]: raise ValueError("model name not exist")
        if model_name == "PTR_COV2": model_path = "/home/alta/summary/pm574/pointer_summarizer/lib/trained_models/PTR_COV2/iter330000.pt"
        self.model = Model(model_path, is_eval=True)

    def sort_beams(self, beams):
        return sorted(beams, key=lambda h: h.avg_log_prob, reverse=True)

    def decode(self, batches, vocab, use_cuda=False):
        pgn_summaries = []
        for batch in batches:
            best_summary = self.beam_search(batch, vocab, use_cuda)

            # Extract the output ids from the hypothesis and convert back to words
            output_ids = [int(t) for t in best_summary.tokens[1:]]
            decoded_words = data.outputids2words(output_ids, vocab, (batch.art_oovs if config.pointer_gen else None))

            # Remove the [STOP] token from decoded_words, if necessary
            try:
                fst_stop_idx = decoded_words.index(data.STOP_DECODING)
                decoded_words = decoded_words[:fst_stop_idx]
            except ValueError:
                decoded_words = decoded_words

            pgn_summary = TreebankWordDetokenizer().detokenize(decoded_words)
            pgn_summaries.append(pgn_summary)

        return pgn_summaries

    def beam_search(self, batch, vocab, use_cuda=False):
        # beam_search_decoding
        enc_batch = batch.enc_batch
        enc_padding_mask = batch.enc_padding_mask
        enc_lens = batch.enc_lens
        enc_batch_extend_vocab = batch.enc_batch_extend_vocab
        extra_zeros = batch.extra_zeros
        c_t_0 = batch.c_t_1
        coverage_t_0 = batch.coverage

        encoder_outputs, encoder_feature, encoder_hidden = self.model.encoder(enc_batch, enc_lens)
        s_t_0 = self.model.reduce_state(encoder_hidden)

        dec_h, dec_c = s_t_0 # 1 x 2*hidden_size
        dec_h = dec_h.squeeze(0)
        dec_c = dec_c.squeeze(0)

        # decoder batch preparation, it has beam_size example initially everything is repeated
        beams = [Beam(tokens=[vocab.word2id(data.START_DECODING)],
                      log_probs=[0.0],
                      state=(dec_h[0], dec_c[0]),
                      context = c_t_0[0],
                      coverage=(coverage_t_0[0] if config.is_coverage else None))
                 for _ in range(config.beam_size)]

        results = []
        steps = 0
        while steps < config.max_dec_steps and len(results) < config.beam_size:
            latest_tokens = [h.latest_token for h in beams]
            latest_tokens = [t if t < vocab.size() else vocab.word2id(data.UNKNOWN_TOKEN) \
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

            final_dist, s_t, c_t, attn_dist, p_gen, coverage_t = self.model.decoder(y_t_1, s_t_1,
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
                if h.latest_token == vocab.word2id(data.STOP_DECODING):
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

        return beams_sorted[0]

def test():
    pgn = PGNwithCoverage("PTR_COV2")
    tokenizer = PGNTokenizer()
    batches = tokenizer.get_decoding_batches_from_docs(docs=[LONG_BORING_TENNIS_ARTICLE, LONG_BORING_TENNIS_ARTICLE],
                                                        beam_size=4,use_cuda=True )

    pgn_summaries = pgn.decode(batches=batches, vocab=tokenizer.vocab, use_cuda=True)

test()
