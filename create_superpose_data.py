import json
import numpy as np
import pdb
from collections import Counter

q_file = "/home/akalra1/projects/adversarial-attacks/data/vqa_v1/OpenEnded_mscoco_train2014_questions.json"
a_file = "/home/akalra1/projects/adversarial-attacks/data/vqa_v1/mscoco_train2014_annotations.json"

def most_common(lst):
    data = Counter(lst)
    return max(lst, key=data.get)

with open(a_file) as json_file:
    a_data = json.load(json_file)

with open(q_file) as json_file:
    q_data = json.load(json_file)


n_ques = len(q_data['questions'])

q_dict = {}
a_dict = {}
m_len = 0
lens = []
two_len = 0
for i in range(n_ques):
    print("Reading " +str(i) +" / " +str(n_ques))
    q = q_data['questions'][i]['question'].lower()
    if q in q_dict.keys():
        q_dict[q].append(q_data['questions'][i])
        a_dict[q].append(a_data['annotations'][i])
        if len(q_dict[q]) == 2:
            two_len += 1
        if len(q_dict[q]) > m_len:
            m_len = len(q_dict[q])
            max_q = q
    else:
        q_dict[q] = [q_data['questions'][i]]
        a_dict[q] = [a_data['annotations'][i]]

for key in q_dict.keys():
    lens.append(len(q_dict[key]))

pdb.set_trace()

q_data1 = q_data
q_data2 = q_data
q_data1['questions'] = []
q_data2['questions'] = []


a_data1 = a_data
a_data2 = a_data
a_data1['annotations'] = []
a_data2['annotations'] = []

for q in q_dict.keys()
    n = len(q_dict[q])
    if n < 2:
       continue 
    else:
       for i in range(n):
            j = i + 1
            a_list1 = [a['answer'] for a in a_dict[q][i]['answers']]
            ans1 = most_common(a_list1)
            while j < n:
                a_list2 = [a['answer'] for a in a_dict[q][j]['answers']]
                ans2 = most_common(a_list2)
            if ans1 != ans2:    #Different answer, add pair to list. Think about adding counts here later ##############################################################################
                q_data1.append(q_dict[q][i])
                q_data2.append(q_dict[q][j])
                a_data1.append(a_dict[q][i])
                a_data2.append(a_dict[q][j])
            else:    #Same answer, skip this pair
                continue
            j += 1


print("Continue to write json files")
pdb.set_trace()

with open('q_pair1.json', 'w') as outfile:  
    json.dump(q_data1, outfile)

with open('q_pair2.json', 'w') as outfile:
    json.dump(q_data2, outfile)

with open('a_pair1.json', 'w') as outfile:
    json.dump(a_data1, outfile)

with open('a_pair2.json', 'w') as outfile:
    json.dump(a_data2, outfile)

