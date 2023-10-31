import json
import torch
from transformers import BertTokenizerFast
from torch.utils.data import Dataset
from typing import List, Dict
import random
def read_json(json_path):
    with open(json_path,'r',encoding='utf-8') as f:
        str=f.read()
        data=json.loads(str)
        return data
def read_txt(txt_path):
    with open(txt_path,'r',encoding='utf-8') as f:
        content=f.readlines()
    for i in range(len(content)):
        content[i]=content[i].replace('\n','')
    return content
#sentence_path : Init_corpus.txt
class Knowledge_Dataset(Dataset):
    def __init__(self, sentence_path):
        self.sentence_path = sentence_path
        self.sentence_corpus = read_txt(self.sentence_path)


    def __len__(self):
        return len(self.sentence_corpus)

    def __getitem__(self, index):
        nega_index=random.randint(0,len(self.sentence_corpus)-1)
        while nega_index == index:
            nega_index=random.randint(0,len(self.sentence_corpus)-1)
        sample = {
            'sentence': self.sentence_corpus[index],
            'negative':self.sentence_corpus[nega_index]
        }

        return sample
def collate_fn(batch: List[Dict[str, any]],tokenizer=BertTokenizerFast.from_pretrained(r'/home/sirui/WMM/Car/model/Encoder/BERT_BASE')) -> Dict[str, torch.Tensor]:
    # 获取每个样本中的句子
    sentences = [example["sentence"] for example in batch]
    negative_sentences=[example["negative"] for example in batch]
    cl_input_ids=[]
    cl_attentionmask=[]
    cl_tokentypeids=[]
    # 将句子转换为tokens

    tokens = tokenizer(sentences,truncation=True,padding='max_length',return_tensors='pt',max_length=512)
    input_ids = tokens['input_ids'].cpu().numpy().tolist()
    attention_mask = tokens['attention_mask'].cpu().numpy().tolist()
    token_type_ids = tokens['token_type_ids'].cpu().numpy().tolist()
    negative_tokens = tokenizer(negative_sentences,truncation=True,padding='max_length',return_tensors='pt',max_length=512)
    ne_input_ids = negative_tokens['input_ids'].cpu().numpy().tolist()
    ne_attention_mask = negative_tokens['attention_mask'].cpu().numpy().tolist()
    ne_token_type_ids = negative_tokens['token_type_ids'].cpu().numpy().tolist()
    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    # indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    for i in range(len(input_ids)):
        cl_input_ids.append([input_ids[i],input_ids[i],ne_input_ids[i]])
        cl_attentionmask.append([attention_mask[i],attention_mask[i],ne_attention_mask[i]])
        cl_tokentypeids.append([token_type_ids[i],token_type_ids[i],ne_token_type_ids[i]])
        pre=i
    tokens['cl_input_ids']=torch.tensor(cl_input_ids)
    tokens['cl_attention_mask']=torch.tensor(cl_attentionmask)
    tokens['cl_token_type_ids']=torch.tensor(cl_tokentypeids)
    return tokens
