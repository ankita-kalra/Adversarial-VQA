import argparse
import os, sys, json
from colour import Color

qtypes = {}

def evaluate_result(qtypes, args):
    resultfile = args.resultfile
    wrong_answers = {'Y/N': [], 'Number': [], 'Color': [], 'Others': []}
    correct_answers = {'Y/N': [], 'Number': [], 'Color': [], 'Others': []}
    print('File: ', resultfile)
    with open(resultfile, 'r') as file:
        lines = file.readlines()
        # For example : lines = ["Wrong!!! , in not a lot of words why would the <unk> be riding up the mountain for the skier  , for safety , skiing"
        line_num = "0"
        for line in lines:
            actual_ans = line.split(',')[-2]
            predicted_ans = line.split(',')[-1]
            if args.predict_by_ans:
                if actual_ans.strip() != predicted_ans.strip():
                    wrong_answers[qtypes[line_num]].append(line_num)
                elif actual_ans.strip() == predicted_ans.strip():
                    correct_answers[qtypes[line_num]].append(line_num)
                else:
                    print('Error')
                    exit()
            else:
                ans = line.split('!')[0]
                if ans == 'Wrong':
                    wrong_answers[qtypes[line_num]].append(line_num)
                elif ans == 'Correct':
                    correct_answers[qtypes[line_num]].append(line_num)
            line_num = str(int(line_num) + 1)

    print('Wrong count:\n', [(i, len(v)) for i, v in wrong_answers.items()])
    print('Correct count:\n', [(i, len(v)) for i, v in correct_answers.items()])
    print('Accuracies:')
    for i in wrong_answers.keys():
        wrong_count = len(wrong_answers[i])
        correct_count = len(correct_answers[i])
        if wrong_count + correct_count > 0:
            acc = correct_count / (wrong_count + correct_count)
        else:
            acc = 'N/A'
        print(i, acc)


def assign_category(orig_file_path):
    line_num = 0
    with open(orig_file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            actual_answer = line.split(',')[-2].strip()
            if actual_answer.isdigit():
                qtypes[line_num] = 'Number'
            elif actual_answer in ['yes', 'no', 'Yes', 'No']:
                qtypes[line_num] = 'Y/N'
            elif check_color(actual_answer):
                qtypes[line_num] = 'Color'
            else:
                qtypes[line_num] = 'Others'
            line_num += 1

    with open('qtypes.json', 'w') as qfile:
        json.dump(qtypes, qfile)

def check_color(inp):
    col = inp.replace(" ", "")
    if isColor(col):
        return True
    elif len(inp.split(' ')) > 1:
        for c in inp.split(' '):
            if isColor(c):
                return True
    return False

def isColor(inp):
    try:
        if Color(inp):
            return True
    except ValueError:
        return False

def parse_arguments():
    parser = argparse.ArgumentParser(description='Parser')
    parser.add_argument('--resultfile', dest='resultfile', type=str, default='')
    parser.add_argument('--typefile', dest='typefile', type=str, default='')
    parser.add_argument('--categorise', dest='categorise', type=int, default=0)
    parser.add_argument('--predict_by_ans', dest='predict_by_ans', type=int, default=1)
    return parser.parse_args()


def main(args):
    args = parse_arguments()
    if args.categorise:
        orig_file_path = args.typefile
        assign_category(orig_file_path)

    else:
        # read dumped file and load in dictionary
        with open('qtypes.json', 'r') as fp:
            qtypes = json.load(fp)

        evaluate_result(qtypes, args)


if __name__ == '__main__':
    main(sys.argv)