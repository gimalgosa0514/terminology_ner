#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 13:33:56 2024

@author: ki_mimang
"""

import pandas as pd
df = pd.read_pickle("./data/valid.pkl")
tag = [
       "B-AF",
       "B-AM",
       "B-CV",
       "B-DT",
       "B-EV",
       "B-FD",
       "B-LC",
       "B-MT",
       "B-OG",
       "B-PS",
       "B-PT",
       "B-QT",
       "B-TI",
       "B-TM",
       "B-TR",
       "I-AF",
       "I-AM",
       "I-CV",
       "I-DT",
       "I-EV",
       "I-FD",
       "I-LC",
       "I-MT",
       "I-OG",
       "I-PS",
       "I-PT",
       "I-QT",
       "I-TI",
       "I-TM",
       "I-TR",
       "O"
       ]

count_tag = [0 for _ in range(len(tag))]
for i in range(len(df)):
    for j in df["labels"].iloc[i]:
        idx = tag.index(j)
        count_tag[idx] += 1

for tags, count in zip(tag,count_tag):
    print("{} = {}".format(tags, count))
