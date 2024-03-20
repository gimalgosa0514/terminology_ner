# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 19:14:38 2024

@author: gimal
"""



import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import pandas as pd
import torch
import os
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR
from transformers import BertTokenizerFast
from transformers import BertPreTrainedModel, BertConfig
from seqeval.metrics import f1_score, accuracy_score




#트레이너에 들어갈 task
class NERTask(pl.LightningModule):
  #model이랑 arguments 받아서 이니셜라이징
  def __init__(self,model : BertPreTrainedModel,args : BertConfig, num_class = 31,learning_rate=5e-5):
    super().__init__()
    self.hg_bert = model
    self.hg_config = args
    self.tokenizer = BertTokenizerFast.from_pretrained("klue/bert-base")
    self.learning_rate = learning_rate
    # 우리가 구현한 버트 모델 및 Config 불러와줌.

    from bert import BERT, BERT_CONFIG
    my_config = BERT_CONFIG(hg_config = self.hg_config)
    # 우리가 만든 버트 모델
    self.encoder = BERT(my_config)
    self.encoder.copy_weights_from_huggingface(self.hg_bert)

    pooled_dim = self.encoder.pooler.dense.weight.shape[-1]
    self.to_output = nn.Linear(pooled_dim,num_class)
    self.criterion = nn.CrossEntropyLoss()

  def cal_loss_and_perf(self, logits, label, attention_mask, perf_check=False):
      # cross-entropy handle the final dimension specially.
      # final dimension should be compatible between logits and predictions

      ## --- handling padding label parts
      num_labels = logits.shape[-1]

      logits = logits.view(-1, num_labels) # [B*seq_len, logit_dim]
      label  = label.view(-1)              # [B * seq_len] flatten 

      active_mask   = attention_mask.view(-1) == 1
      active_logits = logits[active_mask]

      active_labels = label[active_mask]
      loss = self.criterion(active_logits, active_labels)

      ## torch metric specific performance
      if perf_check == True:
          prob = F.softmax(active_logits, dim=-1)
          acc  = self.accuracy(prob, active_labels)
          f1 = self.f1_score(prob, active_labels)
          perf = {'acc':acc, 'f1':f1 }
      else:
          perf = None

      return loss, perf


  def forward(self,input_ids,attention_mask):
    _,seq_hidden_states, layers_attention_scores = self.encoder(
        input_ids = input_ids,
        token_type_ids = None,
        attention_mask = attention_mask)
   
    
    logits = self.to_output(seq_hidden_states)
    return logits

  def training_step(self,batch,batch_idx):
    input_ids, attention_mask, labels = batch
    logits = self(input_ids,attention_mask)
    loss,_ = self.cal_loss_and_perf(logits, labels, attention_mask)
    #로그에 찍음.
    self.log("loss", loss, prog_bar = True, logger= True, on_step=True, on_epoch = True)

    return loss


  def test_step(self,batch,batch_idx):
    input_ids, attention_mask, labels = batch  
    logits = self(input_ids,attention_mask)
    preds = logits.argmax(dim = -1)
    all_token = []
    all_token_predictions = []
    all_token_labels = []
    id2label = [
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
    
    #텍스트 받음
    for tokens in input_ids:
      filtered_token = []
      for i in range(len(tokens)):
        text = self.tokenizer.convert_ids_to_tokens(tokens[i].tolist())

        if text == "[CLS]":
          continue
        elif text == "[PAD]":
          continue
        elif text == "[SEP]":
          filtered_token.append(" ")
        else:
          filtered_token.append(text)
      all_token.append(filtered_token)

    #예측토큰이랑, 실제 정답 토큰 받음.
    token_predictions = preds.detach().cpu().numpy()
    
    for token_prediction, label in zip(token_predictions,labels):
      filtered = []
      filtered_label = []

      for i in range(len(token_prediction)):
        if label[i].tolist() == -100:
          continue
        filtered.append(id2label[token_prediction[i]])
        filtered_label.append(id2label[label[i].tolist()])
        
      all_token_predictions.append(filtered)
      all_token_labels.append(filtered_label)
    
    df = pd.DataFrame({"text": all_token, "labels":all_token_labels, "preds":all_token_predictions})

    #최초 생성이면 w
    if not os.path.exists("test_result.csv"):
      df.to_csv("test_result.csv",mode = "w",index=False,sep = ",")
    else:
      df.to_csv("test_result.csv",mode = "a",index=False,sep = ",")
    test_loss,_ = self.cal_loss_and_perf(logits,labels,attention_mask)
    self.log("test_loss", test_loss, prog_bar = True, logger= True, on_step=True, on_epoch = False)
   
    F1_score = f1_score(all_token_labels, all_token_predictions, average="macro")
    accuracy_sco = accuracy_score(all_token_labels, all_token_predictions)

    return {"loss" : test_loss, "accuracy" : float(accuracy_sco), "F1_score" : float(F1_score) }

  def validation_step(self,batch,batch_idx):
    input_ids, attention_mask, labels = batch

    logits = self(input_ids,attention_mask)

    loss,_ = self.cal_loss_and_perf(logits, labels, attention_mask)
    #로그에 찍음.
    self.log("val_loss", loss, prog_bar = True, logger= True, on_step=True, on_epoch = True)
    return loss

  def configure_optimizers(self):
    optimizer = AdamW(self.parameters(), lr = self.learning_rate)
    scheduler = ExponentialLR(optimizer, gamma=0.9)
    return {
        "optimizer" : optimizer,
        "scheduler" : scheduler,
    }
