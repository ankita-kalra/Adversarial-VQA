import json
import argparse
import os

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Argument Parser for LM')
    parser.add_argument("--train", dest="train", type=str, default="data/train")
    parser.add_argument("--dev", dest="dev", type=str, default="data/val")
    parser.add_argument("--test", dest="test", type=str, default="data/v2_Questions_Test_mscoco/v2_OpenEnded_mscoco_test2015_questions.json")
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
    parser.add_argument("--save_to", dest="save_to", type=str, default=model_dir + '/test_adv_examples')
    return parser.parse_args()




def modify(sentence):
    p_1 = "in not a lot of words "
    p_2 = "in not many words "
    p_3 = "what is the answer to "
    return p_1 + sentence, p_2 + sentence, p_3 + sentence, p_1 + p_3 + sentence + " " + p_2.strip()


def extract_sentences(path, to_save):
    data = []
    for i in range(4):
        data.append(json.load(open(path, 'r')))

    for question_1, question_2, question_3, question_4 in zip(data[0]['questions'], data[1]['questions'],
                                                              data[2]['questions'], data[3]['questions']):
        curr_ques = question_1['question']
        m_1, m_2, m_3, m_all = modify(curr_ques)

        question_1['question'] = m_1
        question_2['question'] = m_2
        question_3['question'] = m_3
        question_4['question'] = m_all

    files = [to_save + '_1.json', to_save + '_2.json', to_save + '_3.json', to_save + '_4.json']
    for i, file in enumerate(files):
        with open(file, 'w') as outfile:
            json.dump(data[i], outfile)


if __name__ == '__main__':
    args = parse_arguments()
    extract_sentences(args.test, args.save_to)