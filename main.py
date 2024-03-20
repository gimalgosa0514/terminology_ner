# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 22:34:19 2024

@author: gimal
"""
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast
from transformers import BertModel, BertConfig


from data_utils import NERDataModule
from utils import Config
from task import NERTask
from trainer import get_trainer

import pytorch_lightning as pl
#from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.callbacks.early_stopping import EarlyStopping



if __name__ == "__main__":
    pl.seed_everything(1234)
    #logger = WandbLogger(name="pl-ner",project="pl-ner")
    #이거 main method행.
    
    #데이터 준비
    tokenizer = BertTokenizerFast.from_pretrained("klue/bert-base")
    dm = NERDataModule(tokenizer=tokenizer)
    dm.prepare_data()
    dm.setup()
    print(dm.train_dataset.label_ids)
    #arguments 준비
    args = Config()

    #label 수
    num_labels = dm.num_class

    #klue/bert-base의 pretrained_model_config 불러오기
    pretrained_model_config = BertConfig.from_pretrained(args.model_name)
    model = BertModel.from_pretrained(args.model_name,num_labels = num_labels)

    #task, trainer 선언
    task = NERTask(model, pretrained_model_config)

    trainer = pl.Trainer(
                            num_sanity_val_steps=0,
                            max_epochs=20, 
                            #strategy = DDPStrategy(find_unused_parameters=True),
                            callbacks=[EarlyStopping(monitor="val_loss", mode="min")],
    #                        logger = logger
                        )

    #학습
    #trainer.fit(task,datamodule=dm)

    #채점
    trainer.test(task,dm.test_dataloader())

