# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 12:17:14 2024

@author: gimal
"""

import os
import json
import random


row_text = []
NE = {}
new_text = {}
i = 0

label = ["AF","AM","CV","DT","EV","FD","LC","MT","OG","PS","PT","QT","TI","TM","TR","O"]
with open("/Users/ki_mimang/Downloads/validation_법령_제개정문_009.json",encoding="utf-8") as f:
    data = json.load(f)
    for data in data["data"]:
        
        for rows in data["rows"]:
            text = rows["text"]
            row_text.append(text)
           
            NE[i] = rows["NE"]
         
            i += 1


for i,text in enumerate(row_text):
    if len(text) > 500 or "+" in text or "-" in text or "│" in text or"|" in text or "┃" in text or "─" in text or "…" in text or len(NE[i]) == 0:
        continue
    TAG = ["O" for _ in range(len(text))]
    c = []
    for j, character in enumerate(text):
        c.append(character)
        for k in range(len(NE[i])): 
            if j == NE[i][k]["begin"]:
                if NE[i][k]["type"] not in label:
                    TAG[j] = "O"
                    break
                else:
                    TAG[j] = "B-"+NE[i][k]["type"]
                    break
            if j < NE[i][k]["end"] and j> NE[i][k]["begin"]:
                if NE[i][k]["type"] not in label:
                    TAG[j] = "O"
                    break
                else:
                    TAG[j] = "I-"+NE[i][k]["type"]
                break
    new_text[i] = {
            "sentence":c,
            "tag": TAG
        }
    print(i)
    print(new_text[i])




"""
for _,v in new_text.items():
  sentence = v["sentence"]
  tag = v["tag"]
  for i in range(len(sentence)):
    with open("./data/train.tsv","a",encoding="utf-8") as f:
      f.write(sentence[i]+"\t"+tag[i]+"\n")
  with open("./data/train.tsv","a",encoding="utf-8") as f:
    f.write("\n\n\n")


"""

key_list = list(new_text.keys())
random.shuffle(key_list)
num_valid = int(len(key_list) * 0.1)
key_valid = key_list[:num_valid]
key_test = key_list[num_valid:num_valid*2]


valid_dict = {k: new_text[k] for k in key_valid}
test_dict = {k: new_text[k] for k in key_test}


for _,v in valid_dict.items():
  sentence = v["sentence"]
  tag = v["tag"]
  for i in range(len(sentence)):
    with open("./data/valid.tsv","a",encoding="utf-8") as f:
      f.write(sentence[i]+"\t"+tag[i]+"\n")
  with open("./data/valid.tsv","a",encoding="utf-8") as f:
    f.write("\n\n\n")
    

for _,v in test_dict.items():
  sentence = v["sentence"]
  tag = v["tag"]
  for i in range(len(sentence)):
    with open("./data/test.tsv","a",encoding="utf-8") as f:
      f.write(sentence[i]+"\t"+tag[i]+"\n")
  with open("./data/test.tsv","a",encoding="utf-8") as f:
    f.write("\n\n\n")
