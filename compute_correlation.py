import os
import csv
import numpy as np

"""
[0] p0: unmodified
[1] p1: in not a lot of words
[2] p2: in not many words
[3] p3: what is the answer to
[4] p1 + p3: in not a lot of words what is the answer to
"""

survey_dir = 'VQA_Study'
stats = {}
for response in os.listdir(survey_dir):
    csv_reader = csv.reader(open(survey_dir + "/" + response), delimiter=',')
    keys = []
    for idx, row in enumerate(csv_reader):
        row = row[2:-1]
        if idx == 0:
            # Determine question type
            for j, col in enumerate(row):
                col = col.lower()
                if 'in not a lot of words what is the answer to' in col:
                    q_key = 'p1_p3_q' + str(j + 1)
                elif 'what is the answer to' in col:
                    q_key = 'p3_q' + str(j + 1)
                elif 'in not many words' in col:
                    q_key = 'p2_q' + str(j + 1)
                elif 'in not a lot of words' in col:
                    q_key = 'p1_q' + str(j + 1)
                else:
                    q_key = 'q' + str(j + 1)
                keys.append(q_key)
        else:
            # Add the answers in corresponding question type
            for j, col in enumerate(row):
                q_key = keys[j]
                if q_key not in stats:
                    stats[q_key] = {}
                if col not in stats[q_key]:
                    stats[q_key][col] = 1
                else:
                    stats[q_key][col] += 1
print(stats)


def count_answers(answers):
    correct = 0
    incorrect = 0
    sorted_by_val = sorted(answers.items(), key=lambda kv: kv[1], reverse=True)
    for i, v in enumerate(sorted_by_val):
        if i == 0:
            correct += v[1]
        else:
            incorrect += v[1]
    return correct, incorrect


def get_corr(unmodified, modified):
    answer_types = {}
    idx = 0
    users_modified = 0
    users_unmodified = 0
    for answer in unmodified:
        users_unmodified += unmodified[answer]
        if answer not in answer_types:
            answer_types[answer] = idx
            idx += 1
        for answer in modified:
            users_modified += modified[answer]
            if answer not in answer_types:
                answer_types[answer] = idx
                idx += 1
    modified_arr = np.zeros(len(answer_types))
    unmodified_arr = np.zeros(len(answer_types))
    for answer in modified:
        modified_arr[answer_types[answer]] = modified[answer] / users_modified
    for answer in unmodified:
        unmodified_arr[answer_types[answer]] = unmodified[answer] / users_unmodified
    return np.corrcoef(unmodified_arr, modified_arr)[1, 0]


# Compute correlation scores now
# q vs p1, q vs p2, q vs p3, q vs p1_p3
corrs = [[], [], [], []]
for q in range(1, 10):
    q_key = 'q' + str(q)
    if q_key in stats:
        for i, prefix in enumerate(['p1_', 'p2_', 'p3_', 'p1_p3_']):
            key = prefix + q_key
            if key in stats:
                corr = get_corr(stats[q_key], stats[key])
                if corr != corr:
                    # Happens when everyone answered same
                    corr = 1
                corrs[i].append(corr)

print('Correlation scores between unmodified question and following prefixes,')
print('in not a lot of words: {:.2f}'.format(np.mean(corrs[0])))
print('in not many words: {:.2f}'.format(np.mean(corrs[1])))
print('what is the answer to: {:.2f}'.format(np.mean(corrs[2])))
print('in not a lot of words what is the answer to: {:.2f}'.format(np.mean(corrs[3])))
