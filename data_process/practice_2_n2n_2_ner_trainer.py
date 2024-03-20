"""
    NLP 101 > N2N Classification (Named Entity Recognition) - Trainer

        - this code is for educational purpose.
        - the code is written for easy understanding not for optimized code.

    Author : Sangkeun Jung (hugmanskj@gmai.com)
    All rights reserved. (2022)
"""

# 전처리 해놓은 데이터를 기준으로 훈련을 할꺼임.
# In this code, we will implement
#   - BERT-based token classification
#   - We will re-use our BERT code
#   - We will simply do character-level named entity recognition
#       - to do this, we will modify tokenization part little bit
#       - checkout how to re-use 'character' level embedding result from the pretrained subpiece embedding
#   - We will import pre-trained BERT weights of huggingface to our BERT
#   - Check how to prepare data and process


## NOTE
##  - if you want to do named entity recognition with sub-piece level
##  - you must pre- and post-process of text and tokenizations. (That is not handled in this lecture)

from types import prepare_class
import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader

PRETRAINED_MODEL_NAME = 'klue/bert-base'


# you can define any type of dataset
# dataset : return an example for batch construction. 
class NERDataset(Dataset):
    """Dataset."""
    def __init__(self, data, max_seq_len, input_pad_id, output_pad_id):
        # 데이터 프레임이 입력으로 들어옴
        self.df = data 
        self.max_seq_len   = max_seq_len
        self.input_pad_id  = input_pad_id
        self.output_pad_id = output_pad_id # -100 !!!!

        def add_pad(x, pad_id):
            T = len(x)
            N = self.max_seq_len - T
            x += [pad_id]*N
            return x 

        def make_mask(x):
            T = len(x)
            N = self.max_seq_len - T
            mask = [1] * T + [0]*N
            return mask 

        ## prepare padding and attention data --> 데이터 프레임은 여기서 이용하는거고 얘는 preprocessing 부분에서 처리해준 거임.
        # 매 하나하나의 시퀀스에 마스크를 씌우는거임. 
        self.masks      = self.df.chars_ids.apply(lambda x :  make_mask(x))
        self.token_ids  = self.df.chars_ids.apply(lambda x :  add_pad(x, self.input_pad_id))
        self.label_ids  = self.df.labels_ids.apply(lambda x : add_pad(x, self.output_pad_id)) # 여기서는 -100이 들어간다고.
        # 중요한건 토큰 개수만큼, 라벨 개수가 들어가고, 입력 토큰이 패딩이다? 아웃풋은 -100
    def __len__(self):
        return len(self.df) 

    def __getitem__(self, idx): 
        item = [
                    # input
                    torch.tensor(self.token_ids[idx]),
                    torch.tensor(self.masks[idx]),

                    # output
                    torch.tensor(self.label_ids[idx])

                ]
        return item


class KLEU_NER_DataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 32, max_seq_len : int=512):
        super().__init__()
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len

    def prepare_data(self):
        # called only on 1 GPU
        fns={
                'train':'./data/ner/train.pkl',
                'valid':'./data/ner/valid.pkl',
                'test':'./data/ner/test.pkl',

                'vocab': './data/ner/label.vocab',
                'vocab_info': './data/ner/vocab.info.json',
                'token_vocab': './data/ner/token.vocab',
            }

        # load pickled files --> 피클파일 로드 하고.
        import pandas as pd
        self.train_data = pd.read_pickle(fns['train']) 
        self.valid_data = pd.read_pickle(fns['valid'])
        self.test_data  = pd.read_pickle(fns['test'])

        #self.train_data = self.train_data[:100]

        print("TRAIN :", len(self.train_data))
        print("VALID :", len(self.valid_data))
        print("TEST  :", len(self.test_data))

        print("----- Train Data Statistics -----")
        # 이걸 프린트 해주는 이유는 흔히 버트나, 트랜스포머 이용할 때 맥시멈 시퀀스 렝스를 정해줘야하는데, 이때 매번 하나씩 하는게 아니라, 
        # describe을 통해 내 데이터의 특징을 보게 되면 쉽게 맥시멈 렝스를 볼 수 있어서 해놓음.
        print(self.train_data.chars.apply(lambda x : len(x)).describe())

        self.label_vocab, self.vocab_info, self.token_vocab = self.load_vocab(fns['vocab'], fns['vocab_info'], fns['token_vocab'])
        self.vocabs = (self.label_vocab, self.vocab_info, self.token_vocab)
        
        # 이게 왜 필요해? 
        # 우리가 패딩에 대한 파트는 계산 시 ignore 할 수 있기 때문에 해준거임.
        self.input_pad_id  = self.vocab_info['input']['pad']
        self.output_pad_id = self.vocab_info['output']['pad'] # -100 for pytorch default

        self.num_class = len(self.label_vocab)


    def load_vocab(self, fn, info_fn, token_fn):
        # load label vocab
        label_vocab = {}
        with open(fn, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.rstrip()
                label, idx = line.split('\t')
                label_vocab[label] = int(idx)

        # load token vocab
        token_vocab = {}
        with open(token_fn, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.rstrip()
                label, idx = line.split('\t')
                token_vocab[label] = int(idx)

        import json
        with open(info_fn, 'r', encoding='utf-8') as f:
            vocab_info = json.load(f)

        return label_vocab, vocab_info, token_vocab

    def setup(self):
        # called on every GPU
        self.train_dataset = NERDataset(self.train_data, self.max_seq_len, self.input_pad_id, self.output_pad_id)
        self.valid_dataset = NERDataset(self.valid_data, self.max_seq_len, self.input_pad_id, self.output_pad_id)
        self.test_dataset  = NERDataset(self.test_data,  self.max_seq_len, self.input_pad_id, self.output_pad_id)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True) # NOTE : Shuffle

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)


import torchmetrics

def result_collapse(outputs, target):
    if len( [x[target] for x in outputs][0].shape ) == 0:
        target_value = torch.stack([x[target] for x in outputs])
    else:
        target_value = torch.cat([x[target] for x in outputs])
    return target_value

class NamedEntityRecognizer(pl.LightningModule): 
    def __init__(self, 
                 num_class, 
                 label_padding_idx,
                 # optiimzer setting
                 learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()  

        ## [text encoder] 
        ## prepare pretrained - TRANSFORMER Model
        from transformers import BertModel, BertConfig
        # 미리 짜놓은 버트 파라미터 웨이트를 카피해주고.
        hg_config = BertConfig.from_pretrained(PRETRAINED_MODEL_NAME)
        hg_bert = BertModel.from_pretrained(PRETRAINED_MODEL_NAME)
    
        from commons.bert import BERT, BERT_CONFIG
        my_config = BERT_CONFIG(hg_config=hg_config)
        self.encoder = BERT(my_config)
        self.encoder.copy_weights_from_huggingface(hg_bert)

        ## BERT has pooler network emplicitly. 
        ## if your transformer does not have pooler, you can use top-layer's specific output.
        pooled_dim = self.encoder.pooler.dense.weight.shape[-1]

        # [to output]
        # 아웃풋에 대한 리니어 만들어주고
        self.to_output = nn.Linear(pooled_dim, num_class) # D -> a single number

        # loss
        self.label_padding_idx = label_padding_idx
        # 여기서 원래 ignore = -100으로 들어가야하지만, 기본적으로 -100으로 되어있기 때문에 따로 넣어주지는 않았음.
        self.criterion = nn.CrossEntropyLoss()  

        # perf
        self.f1_score = torchmetrics.F1(num_classes=num_class)
        self.accuracy = torchmetrics.Accuracy(num_classes=num_class)

    def set_vocabs(self, vocabs):
        self.vocabs = vocabs 

    def forward(self, input_ids, attention_mask):
        # ENCODING with Transformer 
        # 우리가 필요한건 각 스텝마다의 결과이기 때문에 이걸 가져다 쓰는거임.
        _, seq_hidden_states, layers_attention_scores = self.encoder(
                                         input_ids=input_ids, 
                                         token_type_ids=None,
                                         attention_mask=attention_mask
                                     )
        
        # TO CLASS
        logits = self.to_output(seq_hidden_states)
        return logits

    def cal_loss_and_perf(self, logits, label, attention_mask, perf_check=False):
        # cross-entropy handle the final dimension specially.
        # final dimension should be compatible between logits and predictions

        ## --- handling padding label parts
        num_labels = logits.shape[-1]
        
        # 크로스엔트로피는 마지막 디멘젼이 클래스 개수를 가정하게 되는데, 마지막  디멘젼이 여러번 나오니까 나눠서 적용하게 되잖아 그래서 펴버린대.
        # view를 통해서 num_labels 만큼만 남기고 뭉게버림. 그렇게 되면 모든 배치사이즈에 있는 데이터들이 쫙 펼쳐지게 됨.
        logits = logits.view(-1, num_labels) # [B*seq_len, logit_dim] --> 모든 스텝에 대해서 배치 만큼 쫙 펴버리잖아.
        label  = label.view(-1)              # [B * seq_len] flatten 
        

        # logit과 label에서 active_mask만큼 떼준다고 함.
        active_mask   = attention_mask.view(-1) == 1 # 유효한 즉 패딩이 아닌 것만 가져오게 됨.
        # 이걸 이용해서 로짓값을 넣어주게 되면 인덱싱이 된대. --> 평평하게 펴진 logits중에서 active 마스크인 것만 가져오면 그 유요한 토큰에 대한 로짓값과 레이블 값을 가져올 수 있잖아.
        active_logits = logits[active_mask] # 전체 로짓값을 가져오는게 아니라, active_mask된 부분만 가져온다 이말이야.

        active_labels = label[active_mask] # 라벨도 이것처럼 수집 해 줌.
        loss = self.criterion(active_logits, active_labels) #

        ## torch metric specific performance --> 계산 해주는거~
        if perf_check == True:
            prob = F.softmax(active_logits, dim=-1)
            acc  = self.accuracy(prob, active_labels)
            f1 = self.f1_score(prob, active_labels)
            perf = {'acc':acc, 'f1':f1 }
        else:
            perf = None

        return loss, perf

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, label = batch 
        logits = self(input_ids, attention_mask)

        ## loss calculate by flatting sequential data
        loss, _ = self.cal_loss_and_perf(logits, label, attention_mask)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # all logs are automatically stored for tensorboard
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, label = batch 
        logits = self(input_ids, attention_mask)

        result = {
                  'input_ids':input_ids.cpu(),
                  'attention_mask':attention_mask.cpu(),
                  'label':label.cpu(), 
                  'logits':logits.cpu(), 
                 }
        return result

    def validation_epoch_end(self, outputs):
        input_ids  = result_collapse(outputs, 'input_ids').cpu()
        label  = result_collapse(outputs, 'label').cpu()
        attention_mask  = result_collapse(outputs, 'attention_mask').cpu()
        logits = result_collapse(outputs, 'logits').cpu()

        loss, pl_perf = self.cal_loss_and_perf(logits, label, attention_mask, perf_check=True)
        
        ## IOB tag sensitive f1 measure cal --> 매 밸리데이션 파트가 끝날 때 마다 넣어 줌.
        perf = self.cal_iob_perf(input_ids, label, attention_mask, logits, to_fn='./result/ner/pred.valid.result.txt')

        ### get predicted result
        metrics = { f'val_{k}':v for k, v in perf.items() }
        metrics['val_loss'] = loss
        for k, v in pl_perf.items(): metrics[f'pl_{k}'] = v

        self.log_dict(metrics, prog_bar=True, logger=True, on_epoch=True)

    def cal_iob_perf(self, all_input_ids, all_label, all_attention_mask, all_logits, to_fn=None):
        # iob sensitive performance measure
        label_vocab, vocab_info, token_vocab = self.vocabs 
        r_label_vocab = {v:k for k,v in label_vocab.items()}
        r_token_vocab = {v:k for k,v in token_vocab.items()}

        ## make (tokens, reference_classes, predicted_classes) form
        data = []
        for input_id, label, attention_mask, logits in zip(all_input_ids, all_label, all_attention_mask, all_logits):
            #N = (input_id != token_vocab['[PAD]']).sum().item()  # 0 for padding
            N = (attention_mask == 1).sum().item()  # 0 for padding
            input_id = input_id[:N]
            label = label[:N]
            pred = logits.argmax(dim=-1)[:N]
            
            a_sent = []
            for t, r, p in zip(input_id, label, pred):
                t, r, p = t.item(), r.item(), p.item()
                t = r_token_vocab[t]
                if t == '[SEP]': t=' '
                # 데이터인데, 데이터가 실제 토큰, 정답 토큰, 예측된 토큰 형태로 넣어주고 get_perf 해주면 구 단위의 평가 결과를 얻을 수 있음.
                a_sent.append( (t, r_label_vocab[r], r_label_vocab[p]) )
            data.append(a_sent)

        from measure.performance import get_performance
        perf = get_performance(data, to_fn)
        return perf

    def test_step(self, batch, batch_idx):
        input_ids, attention_mask, label = batch 
        logits = self(input_ids, attention_mask)

        result = {
                  'input_ids':input_ids.cpu(),
                  'attention_mask':attention_mask.cpu(),
                  'label':label.cpu(), 
                  'logits':logits.cpu(), 
                 }
        return result

    def test_epoch_end(self, outputs):
        input_ids  = result_collapse(outputs, 'input_ids').cpu()
        label  = result_collapse(outputs, 'label').cpu()
        attention_mask  = result_collapse(outputs, 'attention_mask').cpu()
        logits = result_collapse(outputs, 'logits').cpu()

        loss, pl_perf = self.cal_loss_and_perf(logits, label, attention_mask, perf_check=True)
        
        ## IOB tag sensitive f1 measure cal
        perf = self.cal_iob_perf(input_ids, label, attention_mask, logits, to_fn='./result/ner/pred.test.result.txt')

        ### get predicted result
        metrics = { f'test_{k}':v for k, v in perf.items() }
        metrics['test_loss'] = loss
        for k, v in pl_perf.items(): metrics[f'pl_{k}'] = v
        print(metrics)
        self.log_dict(metrics, logger=True, on_epoch=True)

    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("N2N")
        parser.add_argument('--learning_rate', type=float, default=0.00001)
        return parent_parser


from argparse import ArgumentParser
from pytorch_lightning.callbacks import EarlyStopping
def cli_main():
    pl.seed_everything(1234)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=20, type=int)
    parser.add_argument('--max_seq_length', default=150, type=int)

    parser = pl.Trainer.add_argparse_args(parser)
    parser = NamedEntityRecognizer.add_model_specific_args(parser)
    args = parser.parse_args()

    # ------------
    # data
    # ------------
    dm = KLEU_NER_DataModule.from_argparse_args(args)
    dm.prepare_data()
    dm.setup()
    x = iter(dm.train_dataloader()).next() # <for testing 

    # ------------
    # model
    # ------------
    model = NamedEntityRecognizer(
                                dm.num_class,
                                dm.output_pad_id,
                                args.learning_rate
                           )
    model.set_vocabs(dm.vocabs) # explicitly set (do not include vocabs as hyperparameters)

    # ------------
    # training
    # ------------
    trainer = pl.Trainer(
                            num_sanity_val_steps=0,
                            max_epochs=20, 
                            callbacks=[EarlyStopping(monitor='val_FB1', patience=7, mode='max')],
                            gpus = 1 # if you have gpu -- set number, otherwise zero
                        )
    trainer.fit(model, datamodule=dm)

    # ------------
    # testing
    # ------------
    result = trainer.test(model, test_dataloaders=dm.test_dataloader())


    # ------------
    # Store best model for further processing
    # ------------
    import shutil 
    to_model_fn =  "./result/best_model.ckpt"
    shutil.copyfile(trainer.checkpoint_callback.best_model_path, to_model_fn)
    print("[DUMP] model is dumped at ", to_model_fn)

    # { 'test_FB1': 82.08999633789062,
    #   'test_accuracy': 96.66000366210938,
    #   'test_loss': 0.17168238759040833,
    #   'test_precision': 80.22000122070312,
    #   'test_recall': 84.04000091552734
    #   }

    ## Improvement Directions
    ##  - Transformer + CRF[conditional random field] ( capturing path loss )
    ##  - Transformer fine-tuning techniques 
    ##      - weight initialization 
    ##      - learning rate control
    ##      - batch-size control 
    ##      - ..
    ##      - .. 
    

if __name__ == '__main__':
    cli_main()