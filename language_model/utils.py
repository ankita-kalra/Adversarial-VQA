import torch
import numpy as np
import json


def read_corpus(file_path):
    data = []
    for line in open(file_path):
        sent = line.strip().split(' ')
        sent = ['<s>'] + sent + ['</s>']
        data.append(sent)
    return data


def convert_to_idx(data, vocab):
    idx_data = []
    for each in data:
        idx_data.append([vocab.train.word2id[x] if x in vocab.train.word2id else vocab.train.word2id['<unk>']for x in each])
    return idx_data


def chunk(input, chunk_size, is_evaluation=False):
    if is_evaluation:
        input = np.array([item for sublist in input for item in sublist])
    else:
        ra = np.random.permutation(len(input))
        input = np.array([item for sublist in ra for item in input[sublist]])
    bs = input.shape[0] // chunk_size
    input = input[:bs * chunk_size]
    input = input.reshape((chunk_size, bs)).T
    return input


def extract_sentences(path, to_save):
    file = open(to_save, 'w')
    data = json.load(open(path, 'r'))
    number_of_sentneces = 0
    total_words = 0
    for question in data['questions']:
        file.write(question['question'] + "\n")
        number_of_sentneces += 1
        total_words += len(question['question'].split(' '))
    file.close()
    return number_of_sentneces, total_words / number_of_sentneces

def get_batch_data(data, pos, seq_len):
    seq_len = min(seq_len, len(data) - 1 - pos)
    source = data[pos:pos + seq_len]
    target = data[pos + 1: pos + 1 + seq_len]
    return torch.LongTensor(source), torch.LongTensor(target)


def detach_hidden_state(h):
    if type(h) == torch.autograd.Variable:
        return torch.autograd.Variable(h.data)
    else:
        return tuple(detach_hidden_state(v) for v in h)


def to_tensor(numpy_array):
    # Numpy array -> Tensor
    return torch.from_numpy(numpy_array).float()


def to_variable(tensor, requires_grad=False):
    # Tensor -> Variable (on GPU if possible)
    if torch.cuda.is_available():
        # Tensor -> GPU Tensor
        tensor = tensor.cuda()
    return torch.autograd.Variable(tensor, requires_grad=requires_grad)


