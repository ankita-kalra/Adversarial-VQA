# coding=utf-8
import sys
import numpy as np
from lm_lstm import LM_LSTM
from utils import read_corpus, get_batch_data, convert_to_idx, detach_hidden_state, to_variable, to_tensor, chunk
import torch
import math
import argparse
import os
from vocab import Vocab
from timeit import default_timer as timer


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Argument Parser for LM')
    parser.add_argument("--train", dest="train", type=str, default="data/train")
    parser.add_argument("--dev", dest="dev", type=str, default="data/val")
    parser.add_argument("--test", dest="test", type=str, default="data/test")
    parser.add_argument("--vocab", dest="vocab", type=str, default="data/vocab.bin")

    parser.add_argument("--hidden_dimension", dest="hidden_dimension", type=int, default=256)
    parser.add_argument("--embedding_dimension", dest="embedding_dimension", type=int, default=400)
    parser.add_argument("--n_layers", dest="n_layers", type=int, default=1)
    parser.add_argument("--print_every", dest="print_every", type=int, default=1)
    parser.add_argument("--seed", dest="seed", type=int, default=0)
    parser.add_argument("--learning_rate", dest="learning_rate", type=float, default=0.001)
    parser.add_argument("--num_epochs", dest="num_epochs", type=int, default=30)
    parser.add_argument("--chunk_size", dest="chunk_size", type=int, default=80)
    parser.add_argument("--seq_len", dest="seq_len", type=int, default=32)
    parser.add_argument("--clip_value", dest="clip_value", type=float, default=0)
    parser.add_argument("--freq_cutoff", dest="freq_cutoff", type=int, default=0)
    parser.add_argument("--size", dest="size", type=int, default=50000)
    parser.add_argument("--is_gru", dest="is_gru", type=int, default=2)
    parser.add_argument("--mode", dest="mode", type=int, default=2)
    model_dir_name = 'models'
    parser.add_argument("--generate_len", dest="generate_len", type=int, default=20)
    parser.add_argument("--primer", dest="primer", type=str, default='What is')
    parser.add_argument("--model_dir", dest="model_dir", type=str, default=model_dir_name)
    parser.add_argument("--model_file", dest="model_file", type=str, default='model_interrupt.t7')
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), model_dir_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    return parser.parse_args()


def init_xavier(m):
    if type(m) == torch.nn.Linear:
        fan_in = m.weight.size()[1]
        fan_out = m.weight.size()[0]
        std = np.sqrt(6.0 / (fan_in + fan_out))
        m.weight.data.normal_(0, std)
        if m.bias is not None:
            m.bias.data.zero_()


def get_val_loss(net, loss_fn, valid, seq_len, chunk_size):
    losses = []
    net.eval()
    hidden = net.init_hidden(chunk_size)
    cum_loss = 0
    cum_tgt_words = 0
    for i in range(0, valid.shape[0] - 1, seq_len):
        input_val, label = get_batch_data(valid, i, seq_len)
        predictions, hidden = net(to_variable(input_val), hidden)  # Feed forward
        loss = loss_fn(predictions, to_variable(label).contiguous().view(-1)).sum()  # Compute losses
        losses.append(loss.data.cpu().numpy())
        hidden = detach_hidden_state(hidden)
        cum_loss += loss.data.cpu().numpy()
        tgt_word_num_to_predict = label.size(0) * label.size(1)
        cum_tgt_words += tgt_word_num_to_predict
    return np.asscalar(np.mean(losses)), math.exp(cum_loss / cum_tgt_words)


def train(args):

    # Load the training and dev data
    train_data = read_corpus(args.train)
    dev_data = read_corpus(args.dev)

    # Construct vocab
    vocab = Vocab(train_data, int(args.size), int(args.freq_cutoff))

    train_data = convert_to_idx(train_data, vocab)
    dev_data = convert_to_idx(dev_data, vocab)
    dev = chunk(dev_data, chunk_size=args.chunk_size, is_evaluation=True)

    # Initialize model and its parameters
    model = LM_LSTM(hidden_size=args.hidden_dimension, embedding_dim=args.embedding_dimension,
                    output_size=len(vocab.train), n_layers=args.n_layers, is_gru=args.is_gru)
    model.apply(init_xavier)
    cum_loss = cumulative_tgt_words = 0
    print('Begin Maximum Likelihood training:')

    loss_fn = torch.nn.CrossEntropyLoss(reduce=False)  # loss function / optimizer
    optim = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1.2e-6)
    if torch.cuda.is_available():
        model = model.cuda()
        loss_fn = loss_fn.cuda()

    try:
        prev_loss = 100000
        for epoch in range(1, args.num_epochs + 1):
            start_time = timer()
            losses = []
            minibatch = 0
            train = chunk(train_data, args.chunk_size)
            i = 0
            hidden = model.init_hidden(args.chunk_size)
            while minibatch < train.shape[0] - 2:
                if np.random.random() < 0.95:
                    seq_len = args.seq_len
                else:
                    seq_len = args.seq_len / 2
                seq_len = max(10, int(np.random.normal(seq_len, 5)))
                # Adjusting lr according to seq len
                lr_orig = optim.param_groups[0]['lr']
                optim.param_groups[0]['lr'] = lr_orig * seq_len / args.seq_len
                input_val, label = get_batch_data(train, minibatch, seq_len)
                model.train()
                hidden = detach_hidden_state(hidden)
                optim.zero_grad()  # Reset the gradients
                predictions, hidden = model(to_variable(input_val), hidden)  # Feed forward
                loss = loss_fn(predictions, to_variable(label).contiguous().view(-1)).sum()  # Compute losses
                loss_val = loss.data.cpu().numpy()
                loss = loss / args.chunk_size
                loss.backward()  # Backpropagate the gradients
                if args.clip_value > 0:
                    torch.nn.utils.clip_grad_norm(model.parameters(), args.clip_value)  # Clip gradients
                optim.step()  # Update the network
                minibatch += seq_len
                optim.param_groups[0]['lr'] = lr_orig
                losses.append(loss_val)
                cum_loss += loss_val
                cumulative_tgt_words += label.size(0) * label.size(1)
                sys.stdout.write("[%d/%d] :: Training Avg Loss: %f , Training Avg ppl: %f  \r" % (
                    i, (train.shape[0] - 2) // args.seq_len, np.asscalar(np.mean(losses)), math.exp(
                        cum_loss / cumulative_tgt_words)))
                sys.stdout.flush()
                i += 1
            val_loss, val_ppl = get_val_loss(model, loss_fn, dev, args.seq_len, args.chunk_size)
            if epoch % args.print_every == 0:
                print(
                    "Epoch {} : Training Loss: {:.5f}, Validation loss: {:.5f}, Validation ppl: {:.5f}, Time elapsed {:.2f} mins".
                        format(epoch, np.asscalar(np.mean(losses)), val_loss, val_ppl, (timer() - start_time) / 60))
            if prev_loss > val_loss:
                prev_loss = val_loss
                print("Validation loss decreased...saving model !!!")
                torch.save(model.state_dict(), args.model_dir + '/model.t7'.format(val_loss))
            else:
                print("Validation loss increased...reducing lr !!!")
                optim.param_groups[0]['lr'] /= 4
            cum_loss = cumulative_tgt_words = 0

    except KeyboardInterrupt:
        print("Interrupted...saving model !!!")
        torch.save(model.state_dict(), args.model_dir + '/model_interrupt.t7')


def generate(args):
    # Load the training and dev data
    train_data = read_corpus(args.train)

    # Construct vocab
    vocab = Vocab(train_data, int(args.size), int(args.freq_cutoff))

    # Load the trained model
    model = LM_LSTM(hidden_size=args.hidden_dimension, embedding_dim=args.embedding_dimension,
                    output_size=len(vocab.train), n_layers=args.n_layers, is_gru=args.is_gru)
    if torch.cuda.is_available():
        model = model.cuda()
    model.load_state_dict(torch.load(args.model_dir + '/' + args.model_file))
    model.eval()
    sent = args.primer
    words = []
    for word in sent.split(' '):
        words.append(vocab.train.word2id[word] if word in vocab.train.word2id else vocab.train.word2id['<unk>'])
    var = to_variable(torch.LongTensor(words).unsqueeze(1))

    # Pass it through the model
    hidden = model.init_hidden(1)
    decoded_scores, hidden = model(var, hidden)
    y_hat = decoded_scores[-1, :].data.cpu().numpy().argmax()
    sent += ' ' + vocab.train.id2word[y_hat]
    var = to_variable(torch.LongTensor([int(y_hat)]).unsqueeze(1))
    for _ in range(args.generate_len):
        decoded_scores, hidden = model(var, hidden)
        y_hat = decoded_scores[-1, :].data.cpu().numpy().argmax()
        sent += ' ' + vocab.train.id2word[y_hat]
        var = to_variable(torch.LongTensor([int(y_hat)]).unsqueeze(1))
    print(sent)


def main(args):
    # seed the random number generator (RNG)
    seed = args.seed
    np.random.seed(seed * 13 // 7)
    if args.mode == 0:
        train(args)
    elif args.mode == 1:
        generate(args)
    elif args.mode == 2:
        # compute test perpexlity

        train_data = read_corpus(args.train)

        # Construct vocab
        vocab = Vocab(train_data, int(args.size), int(args.freq_cutoff))

        # Load the trained model
        model = LM_LSTM(hidden_size=args.hidden_dimension, embedding_dim=args.embedding_dimension,
                        output_size=len(vocab.train), n_layers=args.n_layers, is_gru=args.is_gru)
        loss_fn = torch.nn.CrossEntropyLoss(reduce=False)  # loss function / optimizer
        if torch.cuda.is_available():
            model = model.cuda()
            loss_fn = loss_fn.cuda()
        model.load_state_dict(torch.load(args.model_dir + '/' + args.model_file))
        model.eval()
        dev_data = read_corpus(args.test)
        file = open('sentence_ppl.txt', 'w')
        j = 0
        for sentence in dev_data:
            words = []
            for word in sentence:
                words.append(vocab.train.word2id[word] if word in vocab.train.word2id else vocab.train.word2id['<unk>'])
            words = np.array(words).reshape(-1, 1)
            test_loss, test_ppl = get_val_loss(model, loss_fn, words, args.seq_len, 1)
            file.write(' '.join(sentence) + " -> " + str(test_ppl))
            j += 1
            if j > 100:
                break
        file.close()
    elif args.mode == 3:
        # compute test perpexlity

        train_data = read_corpus(args.train)

        # Construct vocab
        vocab = Vocab(train_data, int(args.size), int(args.freq_cutoff))

        # Load the trained model
        model = LM_LSTM(hidden_size=args.hidden_dimension, embedding_dim=args.embedding_dimension,
                        output_size=len(vocab.train), n_layers=args.n_layers, is_gru=args.is_gru)
        loss_fn = torch.nn.CrossEntropyLoss(reduce=False)  # loss function / optimizer
        if torch.cuda.is_available():
            model = model.cuda()
            loss_fn = loss_fn.cuda()
        model.load_state_dict(torch.load(args.model_dir + '/' + args.model_file))
        model.eval()
        dev_data = read_corpus(args.test)
        dev_data = convert_to_idx(dev_data, vocab)
        dev = chunk(dev_data, chunk_size=args.chunk_size, is_evaluation=True)
        test_loss, test_ppl = get_val_loss(model, loss_fn, dev, args.seq_len, args.chunk_size)
        print('Test set perplexity: {}'.format(test_ppl))


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
