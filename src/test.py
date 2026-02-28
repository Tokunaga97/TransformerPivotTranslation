# -*- coding: utf-8 -*-

import torch
from statistics import mean
from train import ReturnModelInfomation
from nltk.translate import bleu_score, gleu_score, chrf_score, nist_score

def translate_sentence(sentence, src_field, trg_field, model, device, max_len=100):
    
    model.eval()

    tokens = sentence
    tokens = [src_field.init_token] + tokens + [src_field.eos_token]
        
    src_indexes = [src_field.vocab.stoi[token] for token in tokens]

    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)
    
    src_mask = model.make_src_mask(src_tensor)
    
    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)

    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]

    for i in range(max_len):

        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)

        trg_mask = model.make_trg_mask(trg_tensor)
        
        with torch.no_grad():
            output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
        
        pred_token = output.argmax(2)[:,-1].item()
        
        trg_indexes.append(pred_token)

        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break
    
    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]
    
    return trg_tokens[1:-1], attention

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
SRC1, TRG1, SRC2, TRG2, test, model1, model2, model_name1, model_name2 = ReturnModelInfomation()

model1.load_state_dict(torch.load(model_name1))
model2.load_state_dict(torch.load(model_name2))

BleuScores = []
GleuScores = []
CHRFScores = []
NISTScores = []

for i in range(len(test.examples)):
    src = vars(test.examples[i])['src']
    trg = vars(test.examples[i])['trg']

    translation1, attention1 = translate_sentence(src, SRC1, TRG1, model1, device)
    translation2, attention2 = translate_sentence(translation1, SRC2, TRG2, model2, device)
    BleuScore = bleu_score.sentence_bleu([translation2], trg, weights = (0.5, 0.5))
    GleuScore = gleu_score.sentence_gleu([translation2], trg)
    CHRFScore = chrf_score.sentence_chrf(translation2, trg)
    NISTScore = nist_score.sentence_nist([translation2], trg)
    BleuScores.append(BleuScore)
    GleuScores.append(GleuScore)
    CHRFScores.append(CHRFScore)
    NISTScores.append(NISTScore)

    print("原文(original sentence):\n" + str(" ".join(src)))
    print("出力訳(output translated sentence)：\n" + str(" ".join(translation2)))
    print("正解(correct sentence)：\n"  + str(" ".join(trg)))
    print("BLEU値(blue score)：" + str(BleuScore))
    print("GLEU値(glue score)：" + str(GleuScore))
    print("CHRF値(chrf score)：" + str(CHRFScore))
    print("NIST値(nist score)：" + str(NISTScore))
print("BLEU値平均(average blue score)："  + str(mean(BleuScores) * 100))
print("GLEU値平均(average glue score)："  + str(mean(GleuScores) * 100))
print("CHRF値平均(average chrf score)："  + str(mean(CHRFScores) * 100))
print("NIST値平均(average nist score)："  + str(mean(NISTScores) * 100))