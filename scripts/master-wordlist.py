import pandas as pd
import numpy as np
import json
import re

data = pd.read_excel('.././data/words-master.xlsx')

keys = data.columns[7:]
word_dict = {}

def get_words(cat_data):
    words = []
    for sent in cat_data.dropna():
        sent = sent.replace('\n',',')
        sent = sent.replace("''",'')
        sent = sent.replace("/",',')
        sent = sent.replace(".",',')
        sent = sent.replace("and",'')
        sent = sent.replace(":",'')
        sent = re.sub(r"[\d]", "", sent)
        for word in sent.split(','):
            word = word.replace("\"", "").replace("\\", "")
            word=word.strip()
            if word == '':
                continue
            if word not in words:
                words.append(word)
    return words

for key in keys:
    word_dict[key] = get_words(data[key])



with open(".././data/word-master-list.json","w") as file:
    json.dump(word_dict,file)