# AdvVQA : Attend, Attack and Destroy

This repository contains the code for our project Adversarial Attacks on VQA Models.

## Visual Attack
###TODO: Add instructions to run the code

## Language Attack
We propose two kinds of language attacks. The code for generating the data files for each of them are located in _language\_attacks/_.

For sub-obj-verb-head replacement attack,
```
python language_attacks/subject_object_replacement.py --train="data/OpenEnded_mscoco_train2014_questions.json" --test="data/OpenEnded_mscoco_val2014_questions.json" --use_glove=0 --parser='stanford-corenlp-full-2017-06-09/stanford-corenlp-full-2017-06-09/'
```
For sub-obj-head replacement attack, use the above command but exlcude the relation _'ROOT'_ in line 125 of _subject_object_replacement.py_.

For the prefix attack, simply use
```
python language_attacks/subject_object_replacement.py --test="data/OpenEnded_mscoco_val2014_questions.json"
```

For the additional experiment of using Glove embeddings for finding similar words, use the following command,
```
python language_attacks/subject_object_replacement.py --train="data/OpenEnded_mscoco_train2014_questions.json" --test="data/OpenEnded_mscoco_val2014_questions.json" --use_glove=1 --glove_embeds="path/to/glove.6B.100d.txt.word2vec" --parser='stanford-corenlp-full-2017-06-09/stanford-corenlp-full-2017-06-09/'
```
