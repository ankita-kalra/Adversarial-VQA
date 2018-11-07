from collections import Counter
from itertools import chain
import pickle
import argparse
from utils import read_corpus, extract_sentences


def parse_arguments():
    parser = argparse.ArgumentParser(
            description='Argument Parser for Vocab')
    parser.add_argument("--freq_cutoff", dest="freq_cutoff", type=int, default=0)
    parser.add_argument("--size", dest="size", type=int, default=50000)
    parser.add_argument("--vocab", dest="vocab", type=str, default="data/vocab.bin")
    return parser.parse_args()


class VocabEntry(object):
    def __init__(self):
        self.word2id = dict()
        self.unk_id = 3
        self.word2id['<pad>'] = 0
        self.word2id['<s>'] = 1
        self.word2id['</s>'] = 2
        self.word2id['<unk>'] = 3

        self.id2word = {v: k for k, v in self.word2id.items()}

    def __getitem__(self, word):
        return self.word2id.get(word, self.unk_id)

    def __contains__(self, word):
        return word in self.word2id

    def __setitem__(self, key, value):
        raise ValueError('vocabulary is readonly')

    def __len__(self):
        return len(self.word2id)

    def __repr__(self):
        return 'Vocabulary[size=%d]' % len(self)

    def id2word(self, wid):
        return self.id2word[wid]

    def add(self, word):
        if word not in self:
            wid = self.word2id[word] = len(self)
            self.id2word[wid] = word
            return wid
        else:
            return self[word]

    def words2indices(self, sents):
        if type(sents[0]) == list:
            return [[self[w] for w in s] for s in sents]
        else:
            return [self[w] for w in sents]

    @staticmethod
    def from_corpus(corpus, size, freq_cutoff=2):
        vocab_entry = VocabEntry()

        word_freq = Counter(chain(*corpus))
        valid_words = [w for w, v in word_freq.items() if v >= freq_cutoff]
        print(f'number of word types: {len(word_freq)}, number of word types w/ frequency >= {freq_cutoff}: {len(valid_words)}')

        top_k_words = sorted(valid_words, key=lambda w: word_freq[w], reverse=True)[:size]
        for word in top_k_words:
            vocab_entry.add(word)

        return vocab_entry


class Vocab(object):
    def __init__(self, sents, vocab_size, freq_cutoff):
        print('initialize vocabulary ..')
        self.train = VocabEntry.from_corpus(sents, vocab_size, freq_cutoff)

    def __repr__(self):
        return 'Vocab(%d words)' % (len(self.train))


if __name__ == '__main__':
    args = parse_arguments()

    # Extracting training and validation data
    sent_no, avg_words = extract_sentences('data/v2_Questions_Train_mscoco/v2_OpenEnded_mscoco_train2014_questions.json', 'data/train')
    print('Train: ')
    print('Total sentence: {}, avg words / sentence: {}'.format(sent_no, avg_words))
    sent_no, avg_words = extract_sentences('data/v2_Questions_Val_mscoco/v2_OpenEnded_mscoco_val2014_questions.json', 'data/val')
    print('Val: ')
    print('Total sentence: {}, avg words / sentence: {}'.format(sent_no, avg_words))
    sent_no, avg_words = extract_sentences('data/v2_Questions_Test_mscoco/v2_OpenEnded_mscoco_test2015_questions.json', 'data/test')
    print('Test: ')
    print('Total sentence: {}, avg words / sentence: {}'.format(sent_no, avg_words))

    print('read in train sentences: %s' % 'data/train')
    sents = read_corpus('data/train')

    vocab = Vocab(sents, int(args.size), int(args.freq_cutoff))
    print('generated vocabulary, %d words' % (len(vocab.train)))

    pickle.dump(vocab, open(args.vocab, 'wb'))
    print('vocabulary saved to %s' % args.vocab)
