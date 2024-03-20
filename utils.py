# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 22:54:46 2024

@author: gimal
"""

import torch
import os
from glob import glob


class Config():
  model_name: str = "klue/bert-base"
  n_epochs: int = 3
  max_seq_len: int = 64,
  batch_size: int = 32 if torch.cuda.is_available() else 4
  learning_rate: float = 5e-5
  adam_epsilon: float = 1e-8
  device: str = "cuda"
  max_grad_norm: float = 1.0
  save_top_k: int = 1
  downstream_model_dir: str ="./pl-ner/0zh1y57k/checkpoints"
  seed: int = 7
  gpu_id:int = 0
  verbose:int = 2
  model_fn: str = "./0.853153_klue_bert-base_16_1.pth"
  infer_batch_size:int = 2
  infer_gpu_id:str =  0
  monitor: str ="min val_loss"
  test_mode: bool =False
  fp16: bool = False
  tpu_cores: int = 0


class DeployConfig():
    def __init__(
            self,
            pretrained_model_name=None,
            downstream_model_dir=None,
            downstream_model_checkpoint_fpath=None,
            downstream_model_labelmap_fpath=None,
            max_seq_length=128,
    ):
        #모델 이름은 인자로 받은 pretrained된 모델 이름.
        self.pretrained_model_name = pretrained_model_name
        #max_seq_length 받아주고
        self.max_seq_length = max_seq_length
        #저장되어있는 체크포인트 경로랑 labelMap (레이블링 구성) 경로 받아줌.
        if downstream_model_checkpoint_fpath is not None and downstream_model_labelmap_fpath is not None:
            self.downstream_model_checkpoint_fpath = downstream_model_checkpoint_fpath
            self.downstream_model_labelmap_fpath = downstream_model_labelmap_fpath
        #아니면 학습된 모델의 경로가 입력된 경우 그 경로에서 받아줌.
        elif downstream_model_dir is not None:
            ckpt_file_names = glob(os.path.join(downstream_model_dir, "*.ckpt"))
            ckpt_file_names = [el for el in ckpt_file_names if "temp" not in el and "tmp" not in el]
            #리스트가 비어있다면 유효하지 않다고 예외 띄우기
            if len(ckpt_file_names) == 0:
                raise Exception(f"downstream_model_dir \"{downstream_model_dir}\" is not valid")
            #들어있다면 맨 뒤의 파일 채택
            selected_fname = ckpt_file_names[-1]
            #min_val_loss를 파일 이름에서 뽑아줌. 이걸로 로스값이 최소인 체크포인트 불러오는거.
            min_val_loss = os.path.split(selected_fname)[-1].replace(".ckpt", "").split("=")[-1].split("-")[0]
            try:
              #비교해서 제일 적은거 가져옴.
                for ckpt_file_name in ckpt_file_names:
                    val_loss = os.path.split(ckpt_file_name)[-1].replace(".ckpt", "").split("=")[-1].split("-")[0]
                    if float(val_loss) < float(min_val_loss):
                        selected_fname = ckpt_file_name
                        min_val_loss = val_loss
            except:
                raise Exception(f"the ckpt file name of downstream_model_directory \"{downstream_model_dir}\" is not valid")
            self.downstream_model_checkpoint_fpath = selected_fname
            self.downstream_model_labelmap_fpath = os.path.join(downstream_model_dir, "label_map.txt")
        else:
            raise Exception("Either downstream_model_dir or downstream_model_checkpoint_fpath must be entered.")
        print(f"downstream_model_checkpoint_fpath: {self.downstream_model_checkpoint_fpath}")
        print(f"downstream_model_labelmap_fpath: {self.downstream_model_labelmap_fpath}")