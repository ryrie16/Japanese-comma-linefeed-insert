from torch import nn
import torch
from pytorch_pretrained_bert import BertForMaskedLM
from transformers import AutoModelForMaskedLM
from attention import Attention
'''
class BertPunc(nn.Module):  
    
    #def __init__(self, segment_size, output_size, dropout):
    def __init__(self, model, model_size, args):
        super(BertPunc, self).__init__()
        #self.bert_vocab_size = 32000
        hidden_size = args['hidden_size']
        self.concat = args['hidden_combine_method'] == 'concat'
        input_size = 768
        #self.bn = nn.BatchNorm1d(segment_size*self.bert_vocab_size)
        #self.fc = nn.Linear(segment_size*self.bert_vocab_size, output_size)
        #self.dropout = nn.Dropout(dropout) 
        self.emb = AutoModelForMaskedLM.from_pretrained(
            "cl-tohoku/bert-base-japanese",
            hidden_dropout_prob=args['hidden_dropout'],
            attention_probs_dropout_prob=args['attention_dropout']
        )
        self.LSTMs = nn.ModuleDict({
            'a': nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=args['num_layers'],
                bidirectional=True,
                batch_first=True,
                dropout=args['dropout'] if args['num_layers'] > 1 else 0
            ),
            'b': nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=args['num_layers'],
                bidirectional=True,
                batch_first=True,
                dropout=args['dropout'] if args['num_layers'] > 1 else 0
            )
            
        })
        
        self.attention_layers = nn.ModuleDict({
            'a': Attention(hidden_size * 2),
            'b': Attention(hidden_size * 2)
        })
        self.dropout = nn.Dropout(dropout)
        
        linear_in_features = hidden_size * 2 if self.concat else hidden_size
        self.Linears = nn.ModuleDict({
            'a': nn.Sequential(
                nn.Linear(linear_in_features, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 2)
            ),
            'b': nn.Sequential(
                nn.Linear(linear_in_features, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 3)
            )
        })
        
        

        
        

    def forward(self, inputs, lens, mask):
        embs = self.emb(inputs, attention_mask=mask)[0]
        #x = torch.tensor(x)
        #x = self.bert(x)
        #x = self.bert(x).logits
        #x = x.view(x.shape[0], -1)
        #x = self.fc(self.dropout(self.bn(x)))
        #return x
        _, (h_a, _) = self.LSTMs['a'](embs)
        if self.concat:
            h_a = torch.cat((h_a[0], h_a[1]), dim=1)
        else:
            h_a = h_a[0] + h_a[1]
        h_a = self.dropout(h_a)

        _, (h_b, _) = self.LSTMs['b'](embs)
        if self.concat:
            h_b = torch.cat((h_b[0], h_b[1]), dim=1)
        else:
            h_b = h_b[0] + h_b[1]
        h_b = self.dropout(h_b)

        logits_a = self.Linears['a'](h_a)
        logits_b = self.Linears['b'](h_b)

        return logits_a, logits_b
'''

class BertPunc(nn.Module):  
    
    def __init__(self, segment_size, output_size1, output_size2, dropout):
        super(BertPunc, self).__init__()
        self.bert = AutoModelForMaskedLM.from_pretrained("cl-tohoku/bert-base-japanese")
        self.bert_vocab_size = 32000
        self.bn1 = nn.BatchNorm1d(segment_size*self.bert_vocab_size)
        self.bn2 = nn.BatchNorm1d(segment_size*self.bert_vocab_size)
        self.fc1 = nn.Linear(segment_size*self.bert_vocab_size, output_size1)
        self.fc2 = nn.Linear(segment_size*self.bert_vocab_size, output_size2)
        self.dropout = nn.Dropout(dropout)
        

    def forward(self, x):  # x = inputs
        #x = torch.tensor(x)
        #x = self.bert(x)
        x = self.bert(x).logits
        x = x.view(x.shape[0], -1)
        x1 = self.fc1(self.dropout(self.bn1(x)))
        x2 = self.fc2(self.dropout(self.bn2(x)))
        return [x1,x2]

'''class BertPunc(nn.Module):  
    
    def __init__(self, segment_size, output_size, dropout):
        super(BertPunc, self).__init__()
        self.bert = AutoModelForMaskedLM.from_pretrained(
            "cl-tohoku/bert-base-japanese",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1
        )
        self.bert_vocab_size = 32000
        #self.bn = nn.BatchNorm1d(segment_size*self.bert_vocab_size)
        #self.fc1 = nn.Linear(segment_size*self.bert_vocab_size, output_size)
        #self.fc2 = nn.Linear(segment_size*self.bert_vocab_size, output_size)
        self.dropout = nn.Dropout(dropout)
        self.LSTMs = nn.ModuleDict({
            'a': nn.LSTM(
                input_size=segment_size*self.bert_vocab_size,
                hidden_size=300,
                num_layers=2,
                bidirectional=True,
                batch_first=True,
                dropout=dropout
            ),
            'b': nn.LSTM(
                input_size=segment_size*self.bert_vocab_size,
                hidden_size=300,
                num_layers=2,
                bidirectional=True,
                batch_first=True,
                dropout=dropout
            )
        })
        
        self.attention_layers=nn.ModuleDict({
            'a':Attention(600),
            'b':Attention(600)
        })
        
        linear_in_features = 600
        
        self.Linears = nn.ModuleDict({
            'a': nn.Sequential(
                nn.Linear(linear_in_features, 300),
                nn.ReLU(),
                nn.Linear(hidden_size, 2)
            ),
            'b': nn.Sequential(
                nn.Linear(linear_in_features, 300),
                nn.ReLU(),
                nn.Linear(hidden_size, 3)
            )
        })
    def forward(self, x):  # x = inputs
        #x = torch.tensor(x)
        bert = self.bert(x)[0]
        #x = self.bert(x)
        _, (h_a, _) = self.LSTMs['a'](bert)
        h_a = torch.cat((h_a[0], h_a[1]), dim=1)
        h_a = self.dropout(h_a)
        
        _, (h_b, _) = self.LSTMs['b'](bert)
        h_b = torch.cat((h_b[0], h_b[1]), dim=1)
        h_b = self.dropout(h_b)
        
        logits_a = self.Linears['a'](h_a)
        logits_b = self.Linears['b'](h_b)

        #x = self.bert(x).logits
        #x = x.view(x.shape[0], -1)
        #x1 = self.fc1(self.dropout(self.bn(x)))
        #x2 = self.fc2(self.dropout(self.bn(x)))
        return logits_a,logits_b
'''