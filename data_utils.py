import torch
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
from transformers import DataCollatorForTokenClassification

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


class NERDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 8, max_seq_len : int=512, tokenizer = None):
        super().__init__()
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.data_collator=self.data_collator_func(tokenizer)
    def prepare_data(self):
        # called only on 1 GPU
        fns={
                'train':'./data/train.pkl',
                'valid':'./data/valid.pkl',
                'test':'./data/test.pkl',
                'vocab': './data/label.vocab',
                'vocab_info': './data/vocab.info.json',
                'token_vocab': './data/token.vocab',
            }

        # load pickled files --> 피클파일 로드 하고.
        import pandas as pd
        self.train_data = pd.read_pickle(fns['train']) 
        self.valid_data = pd.read_pickle(fns['valid'])
        self.test_data = pd.read_pickle(fns['test'])
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
    
    def data_collator_func(self,tokenizer):
        data_collator = DataCollatorForTokenClassification(tokenizer = tokenizer)
        return data_collator
    
    def setup(self,stage=False):
        # called on every GPU
        self.train_dataset = NERDataset(self.train_data, self.max_seq_len, self.input_pad_id, self.output_pad_id)
        self.valid_dataset = NERDataset(self.valid_data, self.max_seq_len, self.input_pad_id, self.output_pad_id)
        self.test_dataset  = NERDataset(self.test_data,  self.max_seq_len, self.input_pad_id, self.output_pad_id)

    def train_dataloader(self):
        
        return DataLoader(self.train_dataset, batch_size=self.batch_size,shuffle=True) # NOTE : Shuffle

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)