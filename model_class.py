import torch
from torch import nn
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.bert.modeling_bert import BertForMaskedLM,BertModel
import os
import json
from transformers import BertModel, BertConfig
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.Tanh()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        return x

class BERTFor_Sencl(nn.Module):

    def __init__(self,check_point):
        super().__init__()
        bert_mlm = BertModel.from_pretrained(check_point)
        #mlm = bert_mlm.cls
        self.config = bert_mlm.config
        self.bert = bert_mlm
        #self.cls=mlm
        self.sim=Similarity(0.05)
        self.sent_num=3
        self.mlp=MLP(768,768)


    def forward(self,
        input_ids = None,
        attention_mask = None,
        token_type_ids = None,
        position_ids = None,
        head_mask = None,
        inputs_embeds = None,
        output_attentions = None,
    ):
        #input_ids----->对比学习输入
        #mlm+****-------> 全词掩码输入
        # Number of sentences in one instance
        # 2: pair instance; 3: pair instance with a hard negative
        num_sent = input_ids.size(1)
        batch_size = input_ids.size(0)
        input_ids = input_ids.view((-1, input_ids.size(-1)))  # (bs * num_sent, len)
        attention_mask = attention_mask.view((-1, attention_mask.size(-1)))  # (bs * num_sent len)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view((-1, token_type_ids.size(-1)))  # (bs * num_sent, len)
        # Get raw embeddings
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            return_dict=True,
        )
        loss_fct = nn.CrossEntropyLoss()
        pooler_output=outputs.last_hidden_state[:,0]
        #breakpoint()
        pooler_output=self.mlp(pooler_output)
        pooler_output=pooler_output.view((batch_size, num_sent, pooler_output.size(-1))) # (bs, num_sent, hidden)
        z1, z2 = pooler_output[:, 0], pooler_output[:, 1]
        z3 = pooler_output[:, 2]
        cos_sim = self.sim(z1.unsqueeze(1), z2.unsqueeze(0))
        if num_sent >= 3:
            z1_z3_cos = self.sim(z1.unsqueeze(1), z3.unsqueeze(0))
            cos_sim = torch.cat([cos_sim, z1_z3_cos], 1)
        labels = torch.arange(cos_sim.size(0)).long().to(device)
        if num_sent == 3:
            # Note that weights are actually logits of weights
            z3_weight = 0
            weights = torch.tensor(
                [[0.0] * (cos_sim.size(-1) - z1_z3_cos.size(-1)) + [0.0] * i + [z3_weight] + [0.0] * (
                            z1_z3_cos.size(-1) - i - 1) for i in range(z1_z3_cos.size(-1))]
            ).to(device)
            cos_sim = cos_sim + weights
        loss = loss_fct(cos_sim, labels)
        #if not return_dict:
        #    output = (cos_sim,) + outputs[2:]
        #    return ((loss,) + output) if loss is not None else output
        return SequenceClassifierOutput(
            loss=loss,
            logits=cos_sim,
            hidden_states=outputs,
            attentions=outputs.attentions,
        )


    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        model = cls(**kwargs)
        state_dict = torch.load(os.path.join(pretrained_model_name_or_path,'pytorch_model.bin'), map_location="cpu")
        model.load_state_dict(state_dict)
        return model

    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        state_dict = self.state_dict()
        torch.save(state_dict, os.path.join(save_directory, "pytorch_model.bin"))
        # config = {"num_labels": self.classifier.out_features, "bert_config": self.bert.config.to_dict()}
        config = {'bert': self.bert.config.to_dict()}
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump(config, f, ensure_ascii=False, indent=4)




