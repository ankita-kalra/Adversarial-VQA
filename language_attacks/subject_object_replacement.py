import json
import argparse
import os
from tqdm import tqdm
from stanfordcorenlp import StanfordCoreNLP
from nltk.corpus import wordnet as wn
import gensim


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Argument Parser for LM')
    parser.add_argument("--train", dest="train", type=str, default="data/v2_Questions_Train_mscoco/v2_OpenEnded_mscoco_train2014_questions.json")
    parser.add_argument("--dev", dest="dev", type=str, default="data/val")
    parser.add_argument("--test", dest="test", type=str, default="data/v2_Questions_Test_mscoco/v2_OpenEnded_mscoco_test2015_questions.json")
    parser.add_argument("--vocab", dest="vocab", type=str, default="data/vocab.bin")
    parser.add_argument("--parser", dest="parser", type=str, default="C:\\Users\\myste\\Downloads\\stanford-corenlp-full-2017-06-09\\stanford-corenlp-full-2017-06-09\\")
    parser.add_argument("--use_glove", dest="use_glove", type=int, default=1)
    parser.add_argument("--glove_embds", dest="glove_embds", type=str, default='C:\\Users\\myste\\Downloads\\glove.6B\\glove.6B.100d.txt.word2vec')
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
    parser.add_argument("--save_to", dest="save_to", type=str, default=model_dir + '/test_adv_examples_replace')
    parser.add_argument("--stopword", dest="stopword", type=str, default='stopword.list')
    return parser.parse_args()


train_vocab = set()
changes = {}


def create_vocab(train_file):
    data = json.load(open(train_file, 'r'))
    for question in data['questions']:
        for word in question['question'][:-1].split(' '):
            if word not in train_vocab:
                train_vocab.add(word)


def pos(tag):
    if tag.startswith('NN'):
        return wn.NOUN
    elif tag.startswith('V'):
        return wn.VERB


def get_synonyms(word, tag):
    lemmas = set()
    lemma_lists = [(ss.lemmas(), ss.max_depth()) for ss in wn.synsets(word, pos(tag))]
    if len(lemma_lists) > 0:
        lemma_lists.sort(key=lambda tup: tup[1])
        lemmas.add(lemma_lists[0][0][0].name())
        #lemmas = [lemma.name() for lemma in sum(lemma_lists, [])]
    return lemmas


def get_similar_word(word, pos_tag):
    for syn in get_synonyms(word, pos_tag):
        # This loop only runs once
        if syn.lower() != word.lower() and syn.lower() in train_vocab:
            return syn.lower()
    return word


def get_similar_word_using_embedding(word, model):
    if word in model.wv and word != '?':
        similar_word = model.wv.most_similar(word, topn=1)[0][0]
        if similar_word in train_vocab:
            return similar_word
    return word


def modify(sentence, nlp, stopwords, relations, model=None):
    parse_tree = nlp.dependency_parse(sentence)
    words = sentence.split(' ')
    found = False
    count = 0
    for i in range(len(parse_tree)):
        node = parse_tree[i]
        if len(node) > 1 and node[0] in relations and node[2] > 0 and node[2] - 1 < len(words) and words[node[2] - 1].lower() not in stopwords:
            if node[0] == 'ROOT':
                pos_tag = wn.VERB
            else:
                pos_tag = wn.NOUN
            if args.use_glove == 0:
                similar_word = get_similar_word(words[node[2] - 1], pos_tag)
            else:
                similar_word = get_similar_word_using_embedding(words[node[2] - 1], model)
            if similar_word != words[node[2] - 1]:
                if words[node[2] - 1] not in changes:
                    changes[words[node[2] - 1]] = similar_word
                found = True
                count += 1
            words[node[2] - 1] = similar_word
    return ' '.join(words), found, count


def extract_sentences(path, to_save, nlp, stopwords, model=None):
    data = json.load(open(path, 'r'))
    questions_changed = 0
    total_changes = 0
    lines = 0
    for question in tqdm(data['questions']):
        curr_ques = question['question']
        curr_ques = curr_ques[:-1] + ' ' + curr_ques[-1]
        modified_question, found, count = modify(curr_ques, nlp, stopwords, ['nsubj', 'dobj', 'ROOT'], model)
        question['question'] = modified_question
        if found:
            questions_changed += 1
            total_changes += count
        lines += 1
        if lines > 100:
            break
    with open(to_save + '.json', 'w') as outfile:
        json.dump(data, outfile)
    return questions_changed, total_changes


if __name__ == '__main__':
    args = parse_arguments()
    print("Loading stanford parser ...")
    nlp = StanfordCoreNLP(args.parser)
    print("Loading stanford parser ... [OK]")
    model = None
    if args.use_glove == 1:
        print("Loading glove embeddings...")
        model = gensim.models.KeyedVectors.load_word2vec_format(args.glove_embds, binary=False)
        print("Loading glove embeddings... [OK]")
    stopwords = set([word.strip() for word in open(args.stopword, 'r').readlines()])
    create_vocab(args.train)
    questions_changed, total_changes = extract_sentences(args.test, args.save_to, nlp, stopwords, model)
    print("Total number of questions changed: {}".format(questions_changed))
    print("Total number of changes in all the questions: {}".format(total_changes))
    print(changes)
