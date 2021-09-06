import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicLSTM(nn.Module):
    '''
    LSTM which can hold variable length sequence, use like TensorFlow's RNN(input, lenght...).
    '''
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=True, dropout=0,
                 bidirectional=False, only_use_last_hidden_state=False, rnn_type='LSTM'):
        super(DynamicLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.only_use_last_hidden_state = only_use_last_hidden_state
        self.rnn_type = rnn_type
        
        if self.rnn_type == 'LSTM':
            self.RNN = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                               bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
        elif self.rnn_type == 'GRU':
            self.RNN = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                              bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
        elif self.rnn_type == 'RNN':
            self.RNN = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                              bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
    
    def forward(self, x, x_len):
        '''
        sequence -> sort -> pad and pack -> process using RNN -> unpack -> unsort
        '''
        '''sort'''
        x_sort_idx = torch.sort(x_len, descending=True)[1].long()
        x_unsort_idx = torch.sort(x_sort_idx)[1].long()
        x_len = x_len[x_sort_idx]
        x = x[x_sort_idx]
        '''pack'''
        x_emb_p = torch.nn.utils.rnn.pack_padded_sequence(x, x_len, batch_first=self.batch_first)
        ''' process '''
        if self.rnn_type == 'LSTM':
            out_pack, (ht, ct) = self.RNN(x_emb_p, None)
        else:
            out_pack, ht = self.RNN(x_emb_p, None)
            ct = None
        '''unsort'''
        ht = ht[:, x_unsort_idx]
        if self.only_use_last_hidden_state:
            return ht
        else:
            out, _ = torch.nn.utils.rnn.pad_packed_sequence(out_pack, batch_first=self.batch_first)
            if self.batch_first:
                out = out[x_unsort_idx]
            else:
                out = out[:, x_unsort_idx]
            if self.rnn_type == 'LSTM':
                ct = ct[:, x_unsort_idx]
            return out, (ht, ct)

class SqueezeEmbedding(nn.Module):
    '''
    Squeeze sequence embedding length to the longest one in the batch
    '''
    def __init__(self, batch_first=True):
        super(SqueezeEmbedding, self).__init__()
        self.batch_first = batch_first
    
    def forward(self, x, x_len):
        '''
        sequence -> sort -> pad and pack -> unpack -> unsort
        '''
        '''sort'''
        x_sort_idx = torch.sort(x_len, descending=True)[1].long()
        x_unsort_idx = torch.sort(x_sort_idx)[1].long()
        x_len = x_len[x_sort_idx]
        x = x[x_sort_idx]
        '''pack'''
        x_emb_p = torch.nn.utils.rnn.pack_padded_sequence(x, x_len, batch_first=self.batch_first)
        '''unpack'''
        out, _ = torch.nn.utils.rnn.pad_packed_sequence(x_emb_p, batch_first=self.batch_first)
        if self.batch_first:
            out = out[x_unsort_idx]
        else:
            out = out[:, x_unsort_idx]
        return out

class SoftAttention(nn.Module):
    '''
    Attention Mechanism for ATAE-LSTM
    '''
    def __init__(self, hidden_dim, embed_dim):
        super(SoftAttention, self).__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.w_h = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.w_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.w_p = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.w_x = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.weight = nn.Parameter(torch.Tensor(hidden_dim+embed_dim))
    
    def forward(self, h, aspect):
        hx = self.w_h(h)
        vx = self.w_v(aspect)
        hv = F.tanh(torch.cat((hx, vx), dim=-1))
        ax = torch.unsqueeze(F.softmax(torch.matmul(hv, self.weight), dim=-1), dim=1)
        rx = torch.squeeze(torch.bmm(ax, h), dim=1)
        hn = h[:, -1, :]
        hs = F.tanh(self.w_p(rx)+self.w_x(hn))
        return hs

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature


    def forward(self, q, k, v, mask=None): # q/k/v.shape: (batchSize, n_head, seqLen, dim)

        attn = torch.matmul(q , k.transpose(-1, -2)) / self.temperature  # attn.shape: (batchSize, n_head, q_seqLen, k_seqLen)
        # attn = torch.matmul(q / self.temperature, k.transpose(1, 2))  # attn.shape: (batchSize, q_seqLen, k_seqLen)

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = F.softmax(attn, dim=-1) # attn.shape: (batchSize, n_head, q_seqLen, k_seqLen)
        output = torch.matmul(attn, v) # output.shape: (batchSize, n_head, q_seqLen, dim)

        return output, attn

# Transformer Parameters
# d_model = 512  # Embedding Size
# d_k = d_v = 64  # dimension of K(=Q), V
n_layers = 8  # number of Encoder of Decoder Layer
# n_heads = 8  # number of heads in Multi-Head Attention

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_heads = 8):
        super(MultiHeadAttention, self).__init__()
        self.d_model, self.d_k, self.d_v, self.n_heads = d_model, d_k, d_v, n_heads
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(d_v * n_heads, d_model, bias=False)
    def forward(self, input_Q, input_K, input_V):
        '''
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        '''

        residual, batch_size = input_Q, input_Q.size(0)
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)
        K = self.W_K(input_K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)
        V = self.W_V(input_V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1,2)

        context, attn = ScaledDotProductAttention(temperature=10)(Q, K, V)
        context = context.transpose(1,2).reshape(batch_size, -1, self.n_heads*self.d_v)
        out = nn.LayerNorm(self.d_model).cuda()(self.fc(context)+residual)

        return out, attn


