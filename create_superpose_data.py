import json
import numpy as np
import pdb

q_file = "/home/akalra1/projects/adversarial-attacks/data/vqa_v1/OpenEnded_mscoco_train2014_questions.json"
a_file = "/home/akalra1/projects/adversarial-attacks/data/vqa_v1/mscoco_train2014_annotations.json"

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
    


