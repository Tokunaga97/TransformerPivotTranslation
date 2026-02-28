# -*- coding: utf-8 -*-

import torch
from nltk.tokenize import word_tokenize
from torchtext import data, datasets
from torchtext.data import BucketIterator



src_lng, mdl_lng, trg_lng = '.pl', '.cs', '.de'
def ExecutePreprocessing(src_lng='.pl', mdl_lng='.cs', trg_lng='.de', batch_size=20, corpus_max=100000):

    def PreprocessingOfDataset(path='JRC-Acquis.cs-pl', path2='cs-pl_train', path3='cs-pl_val', path4='cs-pl_test', src_lng=src_lng, trg_lng=trg_lng, 
                               batch_size=64,val_ratio=0.9, test_ratio=0.95):
        count = corpus_max
        a = count // batch_size
        indexed_n = batch_size * a
       
        path1 = path + src_lng
        with open(path1) as f1:
            l = f1.readlines()
            l = l[:indexed_n]
            fortrain = l[:int((indexed_n * val_ratio))]
            forval = l[int((indexed_n * val_ratio)):]
            fortest = l[int((indexed_n * test_ratio)):]
    
        
        path2_1 = path2 + src_lng
        with open(path2_1, mode='w') as f2:
            for i in range(len(fortrain)):  
                f2.write(str(fortrain[i]) + '\n')
        
        path3_1 = path3 + src_lng
        
        with open(path3_1, 'w') as f4:
            for i in range(len(forval)):
                f4.write(str(forval[i]) + '\n')
    
        path4_1 = path4 + src_lng
        
        with open(path4_1, 'w') as f6:
            for i in range(len(fortest)):
                f6.write(str(fortest[i]) + '\n')
        
        path1 = path + trg_lng
        with open(path1) as f1:
            l = f1.readlines()
            l = l[:indexed_n]
            fortrain = l[:int((indexed_n * val_ratio))]
            forval = l[int((indexed_n * val_ratio)):int((indexed_n * test_ratio))]
            fortest = l[int((indexed_n * test_ratio)):]
    
        path2_2 = path2 + trg_lng
    
        with open(path2_2, mode='w') as f3:
            for i in range(len(fortrain)):
                f3.write(str(fortrain[i]) + '\n')
        
        path3_2 = path3 + trg_lng
            
        with open(path3_2, 'w') as f5:
            for i in range(len(forval)):
                f5.write(str(forval[i]) + '\n')
    
        path4_2 = path4 + trg_lng
            
        with open(path4_2, 'w') as f7:
            for i in range(len(fortest)):
                f7.write(str(fortest[i]) + '\n')
    
    def tokenizer(text):
        return word_tokenize(text)
    
    PreprocessingOfDataset(path='../corpus/JRC-Acquis.cs-pl', path2='../corpus/split/cs-pl_train', path3='../corpus/split/cs-pl_val', path4='../corpus/split/cs-pl_test', src_lng=src_lng, trg_lng=mdl_lng, batch_size=batch_size)
    PreprocessingOfDataset(path='../corpus/JRC-Acquis.cs-de', path2='../corpus/split/cs-de_train', path3='../corpus/split/cs-de_val', path4='../corpus/split/cs-de_test', src_lng=mdl_lng, trg_lng=trg_lng, batch_size=batch_size)
    PreprocessingOfDataset(path='../corpus/JRC-Acquis.de-pl', path2='../corpus/split/de-pl_train', path3='../corpus/split/de-pl_val', path4='../corpus/split/de-pl_test', src_lng=src_lng, trg_lng=trg_lng, batch_size=batch_size,
                           val_ratio=0, test_ratio=0)
    
    SRC1 = data.Field(init_token='<bos>', eos_token='<eos>', lower=True, tokenize=tokenizer, batch_first=True)
    TRG1 = data.Field(init_token='<bos>', eos_token='<eos>', lower=True, tokenize=tokenizer, batch_first=True)
    SRC2 = data.Field(init_token='<bos>', eos_token='<eos>', lower=True, tokenize=tokenizer, batch_first=True)
    TRG2 = data.Field(init_token='<bos>', eos_token='<eos>', lower=True, tokenize=tokenizer, batch_first=True)
    train1, val1, test1 = datasets.TranslationDataset.splits(
        path='../corpus/split/',
        train='cs-pl_train',
        validation='cs-pl_val',
        test='cs-pl_test',
        exts=(src_lng, mdl_lng),
        fields=(SRC1,TRG1)
        )
    
    train2, val2, test2 = datasets.TranslationDataset.splits(
        path='../corpus/split/',
        train='cs-de_train',
        validation='cs-de_val',
        test='cs-de_test',
        exts=(mdl_lng, trg_lng),
        fields=(SRC2,TRG2)
        )
    
    _, _, test = datasets.TranslationDataset.splits(
        path='../corpus/split/',
        train='de-pl_train',
        validation='de-pl_val',
        test='de-pl_test',
        exts=(src_lng, trg_lng),
        fields=(SRC1,TRG2)
        )
    
    SRC1.build_vocab(train1, min_freq=2)
    TRG1.build_vocab(train1, min_freq=2)
    SRC2.build_vocab(train2, min_freq=2)
    TRG2.build_vocab(train2, min_freq=2)
    
    device1 = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device2 = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    train_iter1, valid_iter1, _ = BucketIterator.splits(
        (train1, val1, test1), 
         batch_size = batch_size,
         device = device1)
    
    train_iter2, valid_iter2, _ = BucketIterator.splits(
        (train2, val2, test2), 
         batch_size = batch_size,
         device = device2)
    
    src1_pad_idx = SRC1.vocab.stoi[SRC1.pad_token]
    trg1_pad_idx = TRG1.vocab.stoi[TRG1.pad_token]
    src2_pad_idx = SRC2.vocab.stoi[SRC2.pad_token]
    trg2_pad_idx = TRG2.vocab.stoi[TRG2.pad_token]
    input_dim = len(SRC1.vocab) # the size of vocabulary
    middle_output_dim = len(TRG1.vocab)
    middle_input_dim = len(SRC2.vocab)
    last_output_dim = len(TRG2.vocab)
    
    return train_iter1, valid_iter1, train_iter2, valid_iter2, test, src1_pad_idx, trg1_pad_idx, src2_pad_idx, trg2_pad_idx, input_dim, middle_output_dim, middle_input_dim, last_output_dim

