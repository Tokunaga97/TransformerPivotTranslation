# -*- coding: utf-8 -*-

from statistics import mean
from torch import nn
import torch.utils
from tqdm import tqdm
from preprocessing import ExecutePreprocessing
from model import Encoder, Decoder, Seq2Seq, initialize_weights
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')

def train_phase(model, opt, loss_f, train_iter, clip, device):
    model.train()
    losses = []
    for i, data in enumerate(train_iter):
        x = data.src
        y = data.trg
        opt.zero_grad()

        x = x.to(device)
        y = y.to(device)
        
        output, _ = model(x, y[:,:-1])
        loss = loss_f(output.contiguous().view(-1, output.size(-1)), y[:,1:].contiguous().view(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()
        losses.append(loss.item())
    return mean(losses)

def evaluate_phase(model, loss_f, device, iter):
    model.eval()
    losses = []
    with torch.no_grad():
        
        for i, data in enumerate(iter):
            x = data.src
            y = data.trg

            x = x.to(device)
            y = y.to(device)
        
            output, _ = model(x, y[:,:-1])

            loss = loss_f(output.contiguous().view(-1, output.size(-1)), y[:,1:].contiguous().view(-1))

            losses.append(loss.item())
    return mean(losses)
    


batch_size = 20
max_length = 500
corpus_max = 100000
src_lng, mdl_lng, tgt_lng = '.pl', '.cs', '.de'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

SRC1, TRG1, SRC2, TRG2, train_iter1, valid_iter1, train_iter2, valid_iter2, test, src1_pad_idx, trg1_pad_idx, src2_pad_idx, trg2_pad_idx,\
    input_dim, middle_output_dim, middle_input_dim, last_output_dim = ExecutePreprocessing(src_lng, mdl_lng, tgt_lng, batch_size, corpus_max)
    
hidden_dim = 256 # the dimension of the feedforward network model
enc_layers = 3 # the number of nn.TransformerEncoderLayer 
dec_layers = 3 # the number of nn.TransformerDecoderLayer 
enc_heads = 8 # the number of heads in the multiheadattention models
dec_heads = 8 # the number of heads in the multiheadattention models
enc_pf_dim = 512 # embedding dimension 
dec_pf_dim = 512 # embedding dimension 
enc_dropout = 0.1 # the dropout value
dec_dropout = 0.1 # the dropout value

enc1 = Encoder(input_dim, hidden_dim, enc_layers, enc_heads, enc_pf_dim, 
              enc_dropout, device)
dec1 = Decoder(middle_output_dim, hidden_dim, dec_layers, dec_heads, dec_pf_dim, 
              dec_dropout, device)
enc2 = Encoder(middle_input_dim, hidden_dim, enc_layers, enc_heads, enc_pf_dim, 
              enc_dropout, device)
dec2 = Decoder(last_output_dim, hidden_dim, dec_layers, dec_heads, dec_pf_dim, 
              dec_dropout, device)

model1 = Seq2Seq(enc1, dec1, src1_pad_idx, trg1_pad_idx, device).to(device)
model2 = Seq2Seq(enc2, dec2, src2_pad_idx, trg2_pad_idx, device).to(device)
model1.apply(initialize_weights)
model2.apply(initialize_weights)

learning_rate = 0.0005
opt1 = torch.optim.Adam(model1.parameters(), lr = learning_rate)
opt2 = torch.optim.Adam(model2.parameters(), lr = learning_rate)
loss_f1 = nn.CrossEntropyLoss(ignore_index = trg1_pad_idx)
loss_f2 = nn.CrossEntropyLoss(ignore_index = trg2_pad_idx)
model_name1 = './saved_model/JRC-Acquis_translate_pl-cs'
model_name2 = './saved_model/JRC-Acquis_translate_cs-de' 


def ReturnModelInfomation():
    return (SRC1, TRG1, SRC2, TRG2, test, model1, model2, model_name1, model_name2)
    

all_epoch = 100
clip = 1
best_valid_loss1 = best_valid_loss2 = float('inf')
print("\n")

data_elements = [(model1, opt1, loss_f1, train_iter1, valid_iter1, src_lng, mdl_lng, best_valid_loss1, model_name1),\
                 (model2, opt2, loss_f2, train_iter2, valid_iter2, mdl_lng, tgt_lng, best_valid_loss2, model_name2)
                 ]

for model, opt, loss_f, train_iter, valid_iter, lng1, lng2, best_valid_loss, model_name in data_elements:
    for epoch in tqdm(range(1, all_epoch + 1)):
            
        train_loss = train_phase(model, opt, loss_f, train_iter, clip, device)    
        valid_loss = evaluate_phase(model, loss_f, device, iter=valid_iter)
    
        print("\n[" + str(epoch) + "](train)(" + lng1[1:] + "-" + lng2[1:] +"):"
             + str(train_loss))
        print("[" + str(epoch) + "](valid)(" + lng1[1:] + "-" + lng2[1:] +"):" 
             + str(valid_loss))
    
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), model_name)