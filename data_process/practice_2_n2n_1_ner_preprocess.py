"""
    NLP 101 > N2N Classification (Named Entity Recognition) - Preprocessor

        - this code is for educational purpose.
        - the code is written for easy understanding not for optimized code.

    Author : Sangkeun Jung (hugmanskj@gmai.com)
    All rights reserved. (2022)
"""


in_fns = {
                'train': './data/train.tsv',
                'valid': './data/valid.tsv',
                'test':  './data/test.tsv',

                'token_vocab': './data/token.vocab',
                'label_vocab': './data/label.vocab',
                'vocab_info': './data/vocab.info.json'
        }
# 처리 된 결과물이 여기에 저장이 되는거임. pkl = 피클이라는 파일.
to_fns = {
            'train': './data/train.pkl',
            'valid': './data/valid.pkl',
            'test':  './data/test.pkl',
         }


def load_vocabs(fns):
    # load label vocab
    label_vocab = {}
    with open(fns['label_vocab'], 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip()
            label, idx = line.split('\t')
            label_vocab[label] = int(idx)

    # load token vocab
    token_vocab = {}
    with open(fns['token_vocab'], 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip()
            label, idx = line.split('\t')
            token_vocab[label] = int(idx)

    # vocab info
    import json
    with open(fns['vocab_info'], 'r', encoding='utf-8') as f:
        vocab_info = json.load(f)

    return token_vocab, label_vocab, vocab_info

# prepare global vocabs
token_vocab, label_vocab, vocab_info = load_vocabs(in_fns)


## prepare global label vocab
def load_data(fn):
    # output as pandas
    import pandas as pd 

    chars = []
    labels = []

    a_sent = []
    with open(fn, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip()

            if line == '':
                # end of a sentence
                if len(a_sent) >= 1:
                    chars.append( [x[0] for x in a_sent] )
                    labels.append( [x[1] for x in a_sent] )
                    a_sent = []
                continue
            else:
                fields = line.split('\t')
                char, label = fields[0], fields[-1]
                a_sent.append( (char, label) ) # character level

    data = {
                'chars': chars,
                'labels': labels
    }
    ## pandas
    df = pd.DataFrame(data)
    return df 


def func_chars_to_ids(chars):
    char_ids = []
    for char in chars:
        ## [special processing]
        ## in case of BPE tokenizer, usually, it does not have 
        ## 'space' symbol -- to handle this we just simply use sep token
        # character가 들어오면 char_id가 안에 있으면 그걸 씀. --> 버트 토크나이저 안에는 스페이스에 대한 아이디가 없어서 이렇게 넣어줌.
        if char == ' ': char_id = vocab_info['input']['sep']
        else:           char_id = token_vocab.get(char)

        ## unknown symbol --> 이렇게 처리함.
        if char_id == None:
            char_id = vocab_info['input']['unk']
        char_ids.append( char_id )

    return char_ids

# 특정한 심볼들을 아이디로 바꿔주는 테크닉 즉 토크나이제이션을 매번 할 필요가 없이 한번에 해서 하면 속도가 안떨어짐
# 또는 파이썬의 멀티 프로세스 기능을 이용하여 속도를 빠르게 할 수 있는데 그게 이 부분임.
def do_tokenize(df):
    list_chars = df.chars.tolist()

    # multiprocssing
    from multiprocessing import Pool
    import tqdm
    # 프로세스를 만들어서 하나의 일을 10개가 맡아서 하는거임.
    with Pool(10) as p:
        # 멀티 프로세스가 부르는 함수가 func_char_to_ids임.
        list_chars_ids = list(tqdm.tqdm(p.imap(func_chars_to_ids, list_chars), total=len(list_chars), desc="Tokenization"))
    
    return list_chars_ids

def do_convert_label_to_ids(df):
    list_labels = df.labels.tolist()

    list_labels_ids = []
    for a_sent_labels in list_labels:
        list_labels_ids.append( [ label_vocab.get(label) for label in a_sent_labels ] )

    return list_labels_ids


def do_preprocess(in_fn, to_fn):
    df = load_data(in_fn)
    
    #여기서는 do_tokenizer 먼저 부름.
    list_chars_ids  = do_tokenize(df)
    list_labels_ids = do_convert_label_to_ids(df)

    df['chars_ids']  = list_chars_ids
    df['labels_ids'] = list_labels_ids

    # dump
    # 피클이라는 데이터 타입. 여기서는 pandas data를 pickle datatype으로 변환해 줌.
    df.to_pickle(to_fn)
    print("[DUMP] preprocessed file is dumped at ", to_fn)

if __name__ == '__main__':
    ### (multi) preprocessing
    do_preprocess(in_fns['train'], to_fns['train'])
    do_preprocess(in_fns['valid'], to_fns['valid'])
    do_preprocess(in_fns['test'],  to_fns['test'])




