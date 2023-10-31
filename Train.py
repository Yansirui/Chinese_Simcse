from model_class import BERTFor_Sencl
from Processor import *
from torch.utils.data import DataLoader
from transformers import get_scheduler
from tqdm.auto import tqdm
import torch
import wandb
import argparse
from torch import nn

def train(batch_size,learn):
    #name1=check_point
    batch_size = batch_size
    num_epochs = 3
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    check_point = r'/home/sirui/WMM/Car/model/Encoder/BERT-wwm-ext'
    model=BERTFor_Sencl(check_point=check_point)
    model.train()
    sentence_path=r'/home/sirui/WMM/Medicine/baike_data/SCL_corpus.txt'
    dataset=Knowledge_Dataset(sentence_path=sentence_path)
    dataloader=DataLoader(dataset,batch_size=batch_size,shuffle=True,collate_fn=collate_fn)
    pretrained_params = []
    new_params = []
    for name, param in model.named_parameters():
        if 'bert' in name or 'mlm' in name:
            pretrained_params.append(param)
        else:
            print(name)
            new_params.append(param)
    if learn == 3:
        pretrained_lr = 3e-5
    if learn == 5:
        pretrained_lr = 5e-5
    new_lr=pretrained_lr
    import torch.optim as optim
    optimizer = optim.Adam([
        {'params': pretrained_params, 'lr': pretrained_lr},
        {'params': new_params, 'lr': new_lr}
    ])

    num_training_steps = num_epochs * len(dataloader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    model.to(device)
    progress_bar = tqdm(range(num_training_steps))
    for epoch in range(num_epochs):
        step_num = 0
        for batch in dataloader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            #print(batch['cl_input_ids'].shape)
            outputs=model(input_ids=batch['cl_input_ids'],attention_mask =batch['cl_attention_mask'],token_type_ids=batch['cl_token_type_ids'])
            #2是因为选择2个step更新一次参数，所以除以二防止loss过大引起不好的效果
            loss = outputs.loss / 2
            loss.backward()
            wandb.log({'loss':loss})
            #显存不够时候可以适当增大2，如果显存足够，那就是1个step更新1次，改为1即可，但是针对于该方法而言
            #比如batch_size为8，我想要实现batch_size为16的效果是不可能的，因为batch_size=8时进分类层的维度为[8,16]，也就是每个句子从16个句子中找到positive，然而batch_size=16为[16,32]
            #这种2个step更新一次的方法只是做到了[16,16]而不是[8,32]
            if step_num % 2==0 and step_num!=0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            progress_bar.update(1)
            progress_bar.set_description_str(f'Epoch{epoch}')
            progress_bar.set_postfix(loss=loss.item())
            step_num = step_num+1
        if step_num % 2 != 0:
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            progress_bar.set_description_str(f'Epoch{epoch}')
            progress_bar.set_postfix(loss=loss.item())
    model.bert.save_pretrained(r'/home/sirui/WMM/Medicine/model/Sentence_CL/SCLBERT-b{0}_l{1}'.format(batch_size*2,pretrained_lr))

parser=argparse.ArgumentParser()
parser.add_argument('--batch_size',type=int,help='8 or 16')
parser.add_argument('--learn_rate',type=int,help='3 or 5')
args=parser.parse_args()
batch_size=args.batch_size
learn=args.learn_rate
wandb.init(
    # set the wandb project where this run will be logged
    project='Sentence_CL',
    name='SCLBERT-b{}_l{}'.format(batch_size*2,learn),
    # track hyperparameters and run metadata
    config={
        "learning_rate": 5e-5,
        "architecture": "Transformer",
        "dataset": "Finance",
        "epochs": 3,
    }
)
#print('Project:',project,'   ','Field:',field,'   ','base_model:',model_name)
train(batch_size,learn)
wandb.finish()
