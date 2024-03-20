# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 19:15:41 2024

@author: gimal
"""

import os
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies.ddp import DDPStrategy
def get_trainer(args, return_trainer_only=True):
    #학습한 모델 저장할 경로 지정.
    ckpt_path = os.path.abspath(args.downstream_model_dir)
    os.makedirs(ckpt_path, exist_ok=True)
    #매 epoch마다 체크포인트 저장.
    #min val_loss 상위 1개
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_path,
        save_top_k=args.save_top_k,
        monitor=args.monitor.split()[1],
        mode=args.monitor.split()[0],
        filename='{epoch}-{val_loss:.2f}',
    )
    #트레이너
    trainer = Trainer(
        max_epochs=args.n_epochs,
        fast_dev_run=args.test_mode,
        strategy = DDPStrategy(find_unused_parameters=True),
        num_sanity_val_steps=None if args.test_mode else 0,
        callbacks=[checkpoint_callback],
        default_root_dir=ckpt_path,
        # For GPU Setup
        deterministic=torch.cuda.is_available() and args.seed is not None,
        precision=16 if args.fp16 else 32,

    )
    if return_trainer_only:
        return trainer
    else:
        return checkpoint_callback, trainer
