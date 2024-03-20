# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 15:50:30 2024

@author: gimal
"""

import torch
from transformers import BertConfig, BertModel, BertTokenizerFast
import pandas as pd
from bert import BERT, BERT_CONFIG
from utils import Config, DeployConfig
from task import NERTask

args = Config() 
tokenizer = BertTokenizerFast.from_pretrained(args.model_name)
args = DeployConfig(pretrained_model_name=args.model_name,
                    downstream_model_dir=args.downstream_model_dir,
                    max_seq_length=64)

#학습한 모델 체크포인트 불러오기
fine_tuned_model_ckpt = torch.load(args.downstream_model_checkpoint_fpath,
                                   map_location=torch.device("cpu"))

#모델 파라미터 가져오고
pretrained_model_config = BertConfig.from_pretrained(args.pretrained_model_name,
                                                     num_labels = fine_tuned_model_ckpt["state_dict"]["to_output.bias"].shape.numel())
#torch.Size[13] numel = 13
my_config = BERT_CONFIG(hg_config=pretrained_model_config)
hg_model = BertModel.from_pretrained("klue/bert-base",num_labels = 31)

model = NERTask.load_from_checkpoint("/Users/ki_mimang/Desktop/랩실/프로젝트/terminology_ner/pl-ner/0zh1y57k/checkpoints/epoch=14-step=81960.ckpt"
                                     ,model=hg_model,args = pretrained_model_config,
                                     strict=False)

labels = [
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
#레이블을 알아보기 쉽게 한국어 단어로 변경해줌.
id_to_label = {}
for idx, label in enumerate(labels):
  if "AF" in label:
    label = "인공물"
  elif "AM" in label:
    label = "동물"
  elif "CV" in label:
    label = "제도"
  elif "DT" in label:
    label = "날짜"
  elif "EV" in label:
    label = "사건"
  elif "FD" in label:
    label = "학문 분야"
  elif "LC" in label:
    label = "지역"
  elif "MT" in label:
    label = "물질"
  elif "OG" in label:
    label = "기관"
  elif "PS" in label:
    label = "인물"
  elif "PT" in label:
    label = "식물"
  elif "QT" in label:
    label = "수량"
  elif "TI" in label:
    label = "시간"
  elif "TM" in label:
    label = "용어"
  elif "TR" in label:
    label = "이론"
  else:
    label = "O"
  id_to_label[idx] = label


#인퍼런스 함수.
def inference_fn(sentence):
    tokenn = []
    for i in sentence:
        tokenn.append(i)
        
    inputs = tokenizer(
        [tokenn],
        max_length=args.max_seq_length,
        padding="max_length",
        truncation=True,
        is_split_into_words = True
    )

    with torch.no_grad():
        logits = model(torch.tensor(inputs["input_ids"]),torch.tensor(inputs["attention_mask"]))
        probs = logits[0].softmax(dim=1)
        top_probs, preds = torch.topk(probs, dim=1, k=1)
        print(preds)
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        predicted_tags = [id_to_label[pred.item()] for pred in preds]
        print(predicted_tags)
        result = []
        for token, predicted_tag, top_prob in zip(tokens, predicted_tags, top_probs):
            if token not in [tokenizer.pad_token, tokenizer.cls_token, tokenizer.sep_token]:
                
                token_result = {
                    "토큰": token,
                    "태그": predicted_tag,
                    "확률": str(round(top_prob[0].item(), 4)),
                }
                result.append(token_result)
        df = pd.DataFrame(result)
    return df

