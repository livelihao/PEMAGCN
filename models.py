import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import DynamicLSTM, SqueezeEmbedding, SoftAttention, ScaledDotProductAttention, MultiHeadAttention
import os, sys
import numpy as np
np.set_printoptions(threshold=np.inf)

class LSTM(nn.Module):
    ''' Standard LSTM '''
    def __init__(self, embedding_matrix, opt):
        super(LSTM, self).__init__()
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.lstm = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True)
        self.dense = nn.Linear(opt.hidden_dim, opt.polarities_dim)
    
    def forward(self, inputs):
        text = inputs[0]
        x = self.embed(text)
        x_len = torch.sum(text != 0, dim=-1)
        _, (h_n, _) = self.lstm(x, x_len)
        out = self.dense(h_n[0])
        return out

class AE_LSTM(nn.Module):
    ''' LSTM with Aspect Embedding '''
    def __init__(self, embedding_matrix, opt):
        super(AE_LSTM, self).__init__()
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.squeeze_embedding = SqueezeEmbedding()
        self.lstm = DynamicLSTM(opt.embed_dim*2, opt.hidden_dim, num_layers=1, batch_first=True)
        self.dense = nn.Linear(opt.hidden_dim, opt.polarities_dim)
    
    def forward(self, inputs):
        text, aspect_text = inputs[0], inputs[1]
        x_len = torch.sum(text != 0, dim=-1)
        x_len_max = torch.max(x_len)
        aspect_len = torch.sum(aspect_text != 0, dim=-1).float()
        
        x = self.embed(text)
        x = self.squeeze_embedding(x, x_len)
        aspect = self.embed(aspect_text)
        aspect_pool = torch.div(torch.sum(aspect, dim=1), aspect_len.view(aspect_len.size(0), 1))
        aspect = torch.unsqueeze(aspect_pool, dim=1).expand(-1, x_len_max, -1)
        x = torch.cat((aspect, x), dim=-1)
        
        _, (h_n, _) = self.lstm(x, x_len)
        out = self.dense(h_n[0])
        return out

class ATAE_LSTM(nn.Module):
    ''' Attention-based LSTM with Aspect Embedding '''
    def __init__(self, embedding_matrix, opt):
        super(ATAE_LSTM, self).__init__()
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.squeeze_embedding = SqueezeEmbedding()
        self.lstm = DynamicLSTM(opt.embed_dim*2, opt.hidden_dim, num_layers=1, batch_first=True)
        self.attention = SoftAttention(opt.hidden_dim, opt.embed_dim)
        self.dense = nn.Linear(opt.hidden_dim, opt.polarities_dim)
    
    def forward(self, inputs):
        text, aspect_text = inputs[0], inputs[1]
        x_len = torch.sum(text != 0, dim=-1)
        x_len_max = torch.max(x_len)
        aspect_len = torch.sum(aspect_text != 0, dim=-1).float()
        
        x = self.embed(text)
        x = self.squeeze_embedding(x, x_len)
        aspect = self.embed(aspect_text)
        aspect_pool = torch.div(torch.sum(aspect, dim=1), aspect_len.view(aspect_len.size(0), 1))
        aspect = torch.unsqueeze(aspect_pool, dim=1).expand(-1, x_len_max, -1)
        x = torch.cat((aspect, x), dim=-1)
        
        h, _ = self.lstm(x, x_len)
        hs = self.attention(h, aspect)
        out = self.dense(hs)
        return out

class PBAN(nn.Module):
    ''' Position-aware bidirectional attention network '''
    def __init__(self, embedding_matrix, opt):
        super(PBAN, self).__init__()
        self.text_embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.pos_embed = nn.Embedding(opt.max_length, opt.position_dim)
        self.left_gru = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, 
                                    batch_first=True, bidirectional=True, rnn_type='GRU')
        self.right_gru = DynamicLSTM(opt.embed_dim+opt.position_dim, opt.hidden_dim, num_layers=1, 
                                     batch_first=True, bidirectional=True, rnn_type='GRU')
        self.weight_m = nn.Parameter(torch.Tensor(opt.hidden_dim*2, opt.hidden_dim*2))
        self.bias_m = nn.Parameter(torch.Tensor(1))
        self.weight_n = nn.Parameter(torch.Tensor(opt.hidden_dim*2, opt.hidden_dim*2))
        self.bias_n = nn.Parameter(torch.Tensor(1))
        self.w_r = nn.Linear(opt.hidden_dim*2, opt.hidden_dim)
        self.w_s = nn.Linear(opt.hidden_dim, opt.polarities_dim)
    
    def forward(self, inputs):
        text, aspect_text, position_tag = inputs[0], inputs[1], inputs[2]
        ''' Sentence representation '''
        x = self.text_embed(text)
        # x.shape = [batch size, sen len, embed dim] (64, 80, 300)
        position = self.pos_embed(position_tag)
        # position.shape = [batch size, sen len, pos dim] (64, 80, 100)
        x_len = torch.sum(text != 0, dim=-1)
        x = torch.cat((position, x), dim=-1)
        h_x, _ = self.right_gru(x, x_len)
        # h_x = [batch size, S, num layers * num hiddens] (64, 42, 400)

        ''' Aspect term representation '''
        aspect = self.text_embed(aspect_text)
        # aspect.shape = [batch size, sen len, embed dim] (64, 80, 300)
        aspect_len = torch.sum(aspect_text != 0, dim=-1)
        h_t, _ = self.left_gru(aspect, aspect_len)
        # h_t = [batch size, St, num layers * num hiddens] (64, 8, 400)


        ''' Aspect term to position-aware sentence attention '''
        alpha = F.softmax(F.tanh(torch.add(torch.bmm(torch.matmul(h_t, self.weight_m), torch.transpose(h_x, 1, 2)), self.bias_m)), dim=1)
        # alpha = [batch size, St, S] (64， 8， 42）

        with open('alpha', 'a') as f:
            # f.write(str(self.weight1.squeeze(0).sum(dim=0).cpu().numpy())+'\n')
            # f.write(str(self.weight1.squeeze(0).sum(dim=1).cpu().numpy())+'\n')
            f.write('------------\n')
            f.write('-------------------------------------------------\n')
            alpha_pool = torch.unsqueeze(torch.div(torch.sum(alpha, dim=1), x_len.float().view(x_len.size(0), 1)), dim=1)
            # f.write(str(alpha.cpu().numpy())+'\n')
            f.write(str(alpha)+'\n')
            # f.write(str(alpha.squeeze(0).sum(dim=1).cpu().numpy())+'\n')
            # f.write(str(alpha.squeeze(0).sum(dim=0).cpu().numpy())+'\n')
            # f.write(str(alpha_r.squeeze(0).sum(dim=2).cpu().numpy())+'\n')
            f.write('>>>>>>>>>>>>\n')


        s_x = torch.bmm(alpha, h_x)
        # s_x = [batch size, St, num layers * num hiddens] (64, 8, 400)
        ''' Position-aware sentence attention to aspect term '''
        h_x_pool = torch.unsqueeze(torch.div(torch.sum(h_x, dim=1), x_len.float().view(x_len.size(0), 1)), dim=1)
        # h_x_pool = [batch size, 1, num layers * num hiddens] (64, 1, 400)

        gamma = F.softmax(F.tanh(torch.add(torch.bmm(torch.matmul(h_x_pool, self.weight_n), torch.transpose(h_t, 1, 2)), self.bias_n)), dim=1)
        # gamma = [batch size, 1, St] ( 64, 1, 8)


        h_r = torch.squeeze(torch.bmm(gamma, s_x), dim=1)
        # h_r = [batch size, num layers * num hiddens] (64, 400)
        ''' Output transform '''
        out = F.tanh(self.w_r(h_r))
        # out = [batch size, hidden_dim] (64, 200)
        out = self.w_s(out)
        # out = [batch size, polarities_dim] (64, 3)
        # sys.exit(0)
        return out

class GCAE(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(GCAE, self).__init__()
        self.text_embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.fc_aspect = nn.Linear(opt.embed_dim, 100)
        self.convs1 = nn.ModuleList([nn.Conv1d(opt.embed_dim, 100, fs) for fs in [3,4,5]])
        self.convs2 = nn.ModuleList([nn.Conv1d(opt.embed_dim, 100, fs) for fs in [3,4,5]])
        self.fc = nn.Linear(3*100, 3)

    def forward(self, inputs):
        text, aspect_text = inputs[0], inputs[1]
        x = self.text_embed(text)
        aspect = self.text_embed(aspect_text)
        aspect = aspect.sum(1)/aspect.size(1)

        x1 = [F.tanh(conv(torch.transpose(x, 1, 2))) for conv in self.convs1]
        x2 = [F.relu(conv(torch.transpose(x, 1, 2)) + self.fc_aspect(aspect).unsqueeze(2)) for conv in self.convs2]
        print('x1')
        print(x1[0].shape)
        print('x2')
        print(x2[0].shape)
        x = [i*j for i,j in zip(x1,x2)]
        print('gated')
        print(x[0].shape)
        x0 = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x0 = [i.view(i.size(0), -1) for i in x0]
        input()
        x0 = torch.cat(x0, 1)
        out = self.fc(x0)
        return out

class PGCAE(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(PGCAE, self).__init__()
        self.pos_embed = nn.Embedding(opt.max_length, opt.position_dim)
        self.text_embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.weight_m = nn.Parameter(torch.Tensor(opt.embed_dim, opt.embed_dim))
        self.attention = MultiHeadAttention(d_model=opt.embed_dim, d_k=100, d_v=100)
        self.fc_aspect = nn.Linear(opt.embed_dim, 100)
        self.convs1 = nn.ModuleList([nn.Conv1d(opt.embed_dim+opt.position_dim, 100, fs) for fs in [3,4,5]])
        self.convs2 = nn.ModuleList([nn.Conv1d(opt.embed_dim+opt.position_dim, 100, fs) for fs in [3,4,5]])
        self.convs3 = nn.ModuleList([nn.Conv1d(opt.embed_dim, 100, fs) for fs in [3, 4, 5]])
        self.fc = nn.Linear(3*100, 3)

    def forward(self, inputs):
        text, aspect_text, position_tag = inputs[0], inputs[1], inputs[2]
        x = self.text_embed(text)
        aspect = self.text_embed(aspect_text)
        aspect = aspect.sum(1)/aspect.size(1)
        position = self.pos_embed(position_tag)

        x, att = self.attention(x, x, x)
        x = torch.cat((position, x), dim=-1)

        x1 = [F.tanh(conv(torch.transpose(x, 1, 2))) for conv in self.convs1]
        x2 = [F.relu(conv(torch.transpose(x, 1, 2)) + self.fc_aspect(aspect).unsqueeze(2)) for conv in self.convs2]

        x = [i*j for i,j in zip(x1,x2)]


        x0 = [F.avg_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x0 = [i.view(i.size(0), -1) for i in x0]

        x0 = torch.cat(x0, 1)
        out = self.fc(x0)
        return out

'''# 完整体'''
class TextCNN(nn.Module):
    '''CNN model'''
    def __init__(self, embedding_matrix, opt):
        super(TextCNN, self).__init__()
        self.text_embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.pos_embed = nn.Embedding(opt.max_length, opt.position_dim)
        self.right_gru = DynamicLSTM(opt.embed_dim+opt.position_dim, opt.hidden_dim, num_layers=1,
                                    batch_first=True, bidirectional=True, rnn_type='GRU')
        self.left_gru = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1,
                                    batch_first=True, bidirectional=True, rnn_type='GRU')
        self.weight_m = nn.Parameter(torch.Tensor(opt.hidden_dim*2, opt.hidden_dim*2))
        self.bias_m = nn.Parameter(torch.Tensor(1))
        self.weight_n = nn.Parameter(torch.Tensor(opt.hidden_dim*2, opt.hidden_dim*2))
        self.bias_n = nn.Parameter(torch.Tensor(1))
        self.fc1 = nn.Linear(700, 350)
        self.fc2 = nn.Linear(350, opt.polarities_dim)

        self.convs1 = nn.ModuleList([
                                    nn.Conv1d(in_channels=opt.embed_dim + opt.position_dim,
                                              out_channels=100,
                                              kernel_size=fs,
                                              padding=fs//2)
                                    for fs in [1,1,1]
                                    ])
        self.convs3 = nn.ModuleList([
                                    nn.Conv1d(in_channels=opt.embed_dim + opt.position_dim,
                                              out_channels=100,
                                              kernel_size=fs,
                                              padding=fs//2)
                                    for fs in [3,3,3]
                                    ])
        self.convs5 = nn.ModuleList([
                                    nn.Conv1d(in_channels=opt.embed_dim + opt.position_dim,
                                              out_channels=100,
                                              kernel_size=fs,
                                              padding=fs//2)
                                    for fs in [5,5,5]
                                    ])

        self.weight1 = nn.Parameter(torch.Tensor(100, 100))
        self.bias1 = nn.Parameter(torch.Tensor(1))
        self.weight3 = nn.Parameter(torch.Tensor(100, 100))
        self.bias3 = nn.Parameter(torch.Tensor(1))
        self.weight5 = nn.Parameter(torch.Tensor(100, 100))
        self.bias5 = nn.Parameter(torch.Tensor(1))
        self.weight2 = nn.Parameter(torch.Tensor(400, 300))
        self.bias2 = nn.Parameter(torch.Tensor(1))


    def forward(self, inputs):
        text, aspect_text, position_tag = inputs[0], inputs[1], inputs[2]
        '''Sentence reresentation '''
        x = self.text_embed(text)
        x_len = torch.sum(text != 0, dim=-1)
        # x = [batch_size, sen len, embed dim]
        aspect = self.text_embed(aspect_text) #(bs, 80, 300)
        aspect_len = torch.sum(aspect_text != 0, dim=-1)
        position = self.pos_embed(position_tag)
        # position = [batch_size, sen len, pos dim(100)]
        x = torch.cat((x, position), dim=-1)
        aspect_v = aspect.sum(1) / aspect.size(1) # (bs, 300)

        conved1 = [F.tanh(conv(torch.transpose(x, 1, 2))).squeeze(-1) for conv in self.convs1]
        conved3 = [F.tanh(conv(torch.transpose(x, 1, 2))).squeeze(-1) for conv in self.convs3]
        conved5 = [F.tanh(conv(torch.transpose(x, 1, 2))).squeeze(-1) for conv in self.convs5]
        alpha1 = F.softmax(F.tanh(torch.add(torch.bmm(torch.matmul(torch.transpose(conved1[0], 1, 2), self.weight1), conved1[1]), self.bias1)), dim=1) #(bs,80,80)

        S1 = torch.bmm(alpha1, torch.transpose(conved1[2], 1, 2))#(bs,80,100)
        alpha3 = F.softmax(F.tanh(torch.add(torch.bmm(torch.matmul(torch.transpose(conved3[0], 1, 2), self.weight3), conved3[1]), self.bias3)), dim=1) #(bs,80,80)
        S3 = torch.bmm(alpha3, torch.transpose(conved3[2], 1, 2))#(bs,80,100)
        alpha5 = F.softmax(F.tanh(torch.add(torch.bmm(torch.matmul(torch.transpose(conved5[0], 1, 2), self.weight5), conved5[1]), self.bias5)), dim=1) #(bs,80,80)
        S5 = torch.bmm(alpha5, torch.transpose(conved5[2], 1, 2))#(bs,80,100)

        # S = [torch.transpose(S1,1,2), torch.transpose(S3,1,2), torch.transpose(S5,1,2)]
        S = torch.transpose(torch.cat((S1,S3,S5), dim=2), 1, 2) #(bs,300,80)

        # alpha = F.softmax(F.tanh(torch.add(torch.bmm(torch.matmul(torch.unsqueeze(aspect_v, dim=1), self.weight2), S), self.bias2)), dim=1)
        # pool = torch.bmm(alpha, torch.transpose(S, 1, 2)).squeeze(1)

        x_r, _ = self.right_gru(x, x_len) # (bs, s, 400)
        alpha_r = F.softmax(F.tanh(torch.add(torch.bmm(torch.matmul(x_r, self.weight2), torch.transpose(aspect, 1, 2)), self.bias2)), dim=1) #(bs, s, 80)
        # alpha_r = F.softmax(F.tanh(torch.add(torch.bmm(torch.matmul(aspect, self.weight2), torch.transpose(x_r, 1, 2)), self.bias2)), dim=2)
        # print(alpha_r.shape)

        x_r_2 = torch.transpose(torch.bmm(torch.transpose(alpha_r,1,2), x_r),1,2) #(bs, 400, 80)
        # x_r_2 = torch.bmm(torch.transpose(alpha_r,1,2), x_r) #(bs, 400, 80)
        pool_r = F.max_pool1d(x_r_2, x_r_2.shape[2]).squeeze(2)
        pool = F.avg_pool1d(S, S.shape[2]).squeeze(2) # (bs, 300)

        pool = torch.cat((pool, pool_r), dim=-1)
        out = F.tanh(self.fc1(pool))
        out = self.fc2(out)

        # conved1 = [F.tanh(conv(torch.transpose(x, 1, 2))).squeeze(-1) for conv in self.convs1] # (bs, 100, 80(81))
        # alpha = F.softmax(F.tanh(torch.add(torch.bmm(torch.matmul(torch.transpose(conved1[0], 1, 2), self.weight1), conved1[1]), self.bias1)), dim=1) #(bs,80,80)
        # S = torch.bmm(alpha, torch.transpose(conved1[2], 1, 2))#(bs,80,400)
        # S_pool = torch.unsqueeze(torch.div(torch.sum(S, dim=1), x_len.float().view(x_len.size(0), 1)), dim=1)
        # h_t = [F.tanh(conv(torch.transpose(aspect, 1, 2))) for conv in self.convs1t]
        # # print(h_t[0].shape)
        # h_t = torch.transpose(torch.cat(h_t, dim=1),1,2)
        #
        # alpha2 = F.softmax(F.tanh(torch.add(torch.bmm(torch.matmul(h_t, self.weight_m), torch.transpose(S, 1, 2)), self.bias_m)), dim=1)
        # S_x = torch.bmm(alpha2, S)
        #
        # gamma = F.softmax(F.tanh(torch.add(torch.bmm(torch.matmul(S_pool, self.weight_n), torch.transpose(h_t, 1, 2)), self.bias_n)), dim=1)
        # h_r = torch.squeeze(torch.bmm(gamma, S_x), dim=1)
        # #
        # # pool = F.max_pool1d(S, S.shape[2]).squeeze(2)
        # # cat = torch.cat((pool, pool_t), dim=-1)
        # out = F.tanh(self.fc1(h_r))
        # out = self.fc2(out)

        '''self-attention only x and pos [1,3,5]---- 0.80625'''
        # conved1 = [F.tanh(conv(torch.transpose(x, 1, 2))).squeeze(-1) for conv in self.convs1] # (bs, 100, 80(81))
        # alpha = F.softmax(F.tanh(torch.add(torch.bmm(torch.matmul(torch.transpose(conved1[0], 1, 2), self.weight1), conved1[1]), self.bias1)), dim=1)
        # S = torch.transpose(torch.bmm(alpha, torch.transpose(conved1[2], 1, 2)), 1, 2)
        # pool = F.max_pool1d(S, S.shape[2]).squeeze(2)
        # out = F.tanh(self.fc1(pool))
        # out = self.fc2(out)

        ''' aspect attention x --- 0.7946'''
        # conved = [F.tanh(conv(torch.transpose(x, 1, 2))).squeeze(-1) for conv in self.convs1] # (bs, 100, 80(81))
        # # print(conved[0].shape)
        # conved2 = [F.tanh(conv(torch.transpose(aspect, 1, 2))).squeeze(-1) for conv in self.convs2] # (bs, 100, 80)
        # # print(conved2[0].shape)
        # alpha = [F.softmax(F.tanh(torch.add(torch.bmm(torch.matmul(torch.transpose(i, 1, 2), self.weight1), j), self.bias1)), dim=1)  for j, i in zip(conved, conved2)] # (bs, 80, 80)
        # # print(alpha[0].shape)
        # s = [torch.bmm(a, torch.transpose(conv, 1, 2)) for a, conv in zip(alpha, conved)] # (bs, 80, 100)\
        # s = [torch.transpose(i, 1, 2) for i in s]
        # # print(s[0].shape)
        # pool = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in s] # (bs, 100)
        # # print(pool[0].shape)
        # out = torch.cat(pool, 1)
        # out = F.tanh(self.fc1(out)) # (bs, 300)
        # out = self.fc2(out) # (bs, 150)

        '''only cnn [1,3,5] 0.7982'''
        # pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved1]# (bs, 100)
        # cat = torch.cat(pooled, 1)
        # out = F.tanh(self.fc1(cat))
        # out = self.fc2(out)
        return out

class KimCNN(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(KimCNN, self).__init__()
        self.text_embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.fc1 = nn.Linear(300, 150)
        self.fc2 = nn.Linear(150, opt.polarities_dim)

        self.convs1 = nn.ModuleList([
                                    nn.Conv1d(in_channels=opt.embed_dim,
                                              out_channels=100,
                                              kernel_size=fs)
                                    for fs in [3,4,5]
                                    ])

    def forward(self, inputs):
        text = inputs[0]
        x = self.text_embed(text)
        conved = [F.tanh(conv(torch.transpose(x, 1, 2))).squeeze(-1) for conv in self.convs1]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]# (bs, 100)
        cat = torch.cat(pooled, 1)
        out = F.tanh(self.fc1(cat))
        out = self.fc2(out)
        return out

'''# 对照实验SACN'''
# class TextCNN(nn.Module):
#     '''CNN model'''
#     def __init__(self, embedding_matrix, opt):
#         super(TextCNN, self).__init__()
#         self.text_embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
#         self.pos_embed = nn.Embedding(opt.max_length, opt.position_dim)
#         # self.right_gru = DynamicLSTM(opt.embed_dim+opt.position_dim, opt.hidden_dim, num_layers=1,
#         #                             batch_first=True, bidirectional=True, rnn_type='GRU')
#         # self.left_gru = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1,
#         #                             batch_first=True, bidirectional=True, rnn_type='GRU')
#         # self.weight_m = nn.Parameter(torch.Tensor(opt.hidden_dim*2, opt.hidden_dim*2))
#         # self.bias_m = nn.Parameter(torch.Tensor(1))
#         # self.weight_n = nn.Parameter(torch.Tensor(opt.hidden_dim*2, opt.hidden_dim*2))
#         # self.bias_n = nn.Parameter(torch.Tensor(1))
#         self.fc1 = nn.Linear(300, 150)
#         self.fc2 = nn.Linear(150, opt.polarities_dim)
#
#         self.convs1 = nn.ModuleList([
#                                     nn.Conv1d(in_channels=opt.embed_dim + opt.position_dim,
#                                               out_channels=100,
#                                               kernel_size=fs,
#                                               padding=fs//2)
#                                     for fs in [1,1,1]
#                                     ])
#         self.convs3 = nn.ModuleList([
#                                     nn.Conv1d(in_channels=opt.embed_dim + opt.position_dim,
#                                               out_channels=100,
#                                               kernel_size=fs,
#                                               padding=fs//2)
#                                     for fs in [3,3,3]
#                                     ])
#         self.convs5 = nn.ModuleList([
#                                     nn.Conv1d(in_channels=opt.embed_dim + opt.position_dim,
#                                               out_channels=100,
#                                               kernel_size=fs,
#                                               padding=fs//2)
#                                     for fs in [5,5,5]
#                                     ])
#
#         self.weight1 = nn.Parameter(torch.Tensor(100, 100))
#         self.bias1 = nn.Parameter(torch.Tensor(1))
#         self.weight3 = nn.Parameter(torch.Tensor(100, 100))
#         self.bias3 = nn.Parameter(torch.Tensor(1))
#         self.weight5 = nn.Parameter(torch.Tensor(100, 100))
#         self.bias5 = nn.Parameter(torch.Tensor(1))
#         self.weight2 = nn.Parameter(torch.Tensor(400, 300))
#         self.bias2 = nn.Parameter(torch.Tensor(1))
#
#
#     def forward(self, inputs):
#         text, aspect_text, position_tag = inputs[0], inputs[1], inputs[2]
#         '''Sentence reresentation '''
#         x = self.text_embed(text)
#         x_len = torch.sum(text != 0, dim=-1)
#         # x = [batch_size, sen len, embed dim]
#         # aspect = self.text_embed(aspect_text) #(bs, 80, 300)
#         aspect_len = torch.sum(aspect_text != 0, dim=-1)
#         position = self.pos_embed(position_tag)
#         # position = [batch_size, sen len, pos dim(100)]
#         x = torch.cat((x, position), dim=-1)
#
#         # aspect_v = aspect.sum(1) / aspect.size(1) # (bs, 300)
#
#         conved1 = [F.tanh(conv(torch.transpose(x, 1, 2))).squeeze(-1) for conv in self.convs1]
#         conved3 = [F.tanh(conv(torch.transpose(x, 1, 2))).squeeze(-1) for conv in self.convs3]
#         conved5 = [F.tanh(conv(torch.transpose(x, 1, 2))).squeeze(-1) for conv in self.convs5]
#         alpha1 = F.softmax(F.tanh(torch.add(torch.bmm(torch.matmul(torch.transpose(conved1[0], 1, 2), self.weight1), conved1[1]), self.bias1)), dim=1) #(bs,80,80)
#         S1 = torch.bmm(alpha1, torch.transpose(conved1[2], 1, 2))#(bs,80,100)
#         alpha3 = F.softmax(F.tanh(torch.add(torch.bmm(torch.matmul(torch.transpose(conved3[0], 1, 2), self.weight3), conved3[1]), self.bias3)), dim=1) #(bs,80,80)
#         S3 = torch.bmm(alpha3, torch.transpose(conved3[2], 1, 2))#(bs,80,100)
#         alpha5 = F.softmax(F.tanh(torch.add(torch.bmm(torch.matmul(torch.transpose(conved5[0], 1, 2), self.weight5), conved5[1]), self.bias5)), dim=1) #(bs,80,80)
#         S5 = torch.bmm(alpha5, torch.transpose(conved5[2], 1, 2))#(bs,80,100)
#
#         S = torch.transpose(torch.cat((S1,S3,S5), dim=2),1,2) #(bs,300,80)
#
#
#         # x_r, _ = self.right_gru(x, x_len) # (bs, s, 400)
#         # alpha_r = F.softmax(F.tanh(torch.add(torch.bmm(torch.matmul(x_r, self.weight2), torch.transpose(aspect, 1, 2)), self.bias2)),dim=1) #(bs, s, 80)
#         # x_r_2 = torch.transpose(torch.bmm(torch.transpose(alpha_r,1,2), x_r),1,2) #(bs, 400, 80)
#         # pool_r = F.max_pool1d(x_r_2, x_r_2.shape[2]).squeeze(2)
#         pool = F.avg_pool1d(S, S.shape[2]).squeeze(2) # (bs, 300)
#
#         # pool = torch.cat((pool, pool_r), dim=-1)
#         out = F.tanh(self.fc1(pool))
#         out = self.fc2(out)
#
#         return out

'''# 对照实验RNN-AN'''
# class TextCNN(nn.Module): # 对照实验RNN-AN
#     '''CNN model'''
#     def __init__(self, embedding_matrix, opt):
#         super(TextCNN, self).__init__()
#         self.text_embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
#         self.pos_embed = nn.Embedding(opt.max_length, opt.position_dim)
#         self.right_gru = DynamicLSTM(opt.embed_dim+opt.position_dim, opt.hidden_dim, num_layers=1,
#                                     batch_first=True, bidirectional=True, rnn_type='GRU')
#         self.left_gru = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1,
#                                     batch_first=True, bidirectional=True, rnn_type='GRU')
#         self.weight_m = nn.Parameter(torch.Tensor(opt.hidden_dim*2, opt.hidden_dim*2))
#         self.bias_m = nn.Parameter(torch.Tensor(1))
#         self.weight_n = nn.Parameter(torch.Tensor(opt.hidden_dim*2, opt.hidden_dim*2))
#         self.bias_n = nn.Parameter(torch.Tensor(1))
#         self.fc1 = nn.Linear(400, 200)
#         self.fc2 = nn.Linear(200, opt.polarities_dim)
#         self.weight2 = nn.Parameter(torch.Tensor(400, 300))
#         self.bias2 = nn.Parameter(torch.Tensor(1))
#
#     def forward(self, inputs):
#         text, aspect_text, position_tag = inputs[0], inputs[1], inputs[2]
#         '''Sentence reresentation '''
#         x = self.text_embed(text)
#         x_len = torch.sum(text != 0, dim=-1)
#         # x = [batch_size, sen len, embed dim]
#         aspect = self.text_embed(aspect_text) #(bs, 80, 300)
#         aspect_len = torch.sum(aspect_text != 0, dim=-1)
#         position = self.pos_embed(position_tag)
#         # position = [batch_size, sen len, pos dim(100)]
#         x = torch.cat((x, position), dim=-1)
#
#         x_r, _ = self.right_gru(x, x_len) # (bs, s, 400)
#         alpha_r = F.softmax(F.tanh(torch.add(torch.bmm(torch.matmul(x_r, self.weight2), torch.transpose(aspect, 1, 2)), self.bias2)),dim=1) #(bs, s, 80)
#         x_r_2 = torch.transpose(torch.bmm(torch.transpose(alpha_r,1,2), x_r),1,2) #(bs, 400, 80)
#         pool = F.max_pool1d(x_r_2, x_r_2.shape[2]).squeeze(2)
#
#         # pool = torch.cat((pool, pool_r), dim=-1)
#         out = F.tanh(self.fc1(pool))
#         out = self.fc2(out)
#
#         return out

'''# CAN'''
# class TextCNN(nn.Module): # CAN
#     '''CNN model'''
#     def __init__(self, embedding_matrix, opt):
#         super(TextCNN, self).__init__()
#         self.text_embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
#         self.pos_embed = nn.Embedding(opt.max_length, opt.position_dim)
#         self.right_gru = DynamicLSTM(opt.embed_dim+opt.position_dim, opt.hidden_dim, num_layers=1,
#                                     batch_first=True, bidirectional=True, rnn_type='GRU')
#         self.left_gru = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1,
#                                     batch_first=True, bidirectional=True, rnn_type='GRU')
#         self.weight_m = nn.Parameter(torch.Tensor(opt.hidden_dim*2, opt.hidden_dim*2))
#         self.bias_m = nn.Parameter(torch.Tensor(1))
#         self.weight_n = nn.Parameter(torch.Tensor(opt.hidden_dim*2, opt.hidden_dim*2))
#         self.bias_n = nn.Parameter(torch.Tensor(1))
#         self.fc1 = nn.Linear(700, 350)
#         self.fc2 = nn.Linear(350, opt.polarities_dim)
#
#         self.convs1 = nn.ModuleList([
#                                     nn.Conv1d(in_channels=opt.embed_dim + opt.position_dim,
#                                               out_channels=100,
#                                               kernel_size=fs,
#                                               padding=fs//2)
#                                     for fs in [1,3,5]
#                                     ])
#
#         self.weight2 = nn.Parameter(torch.Tensor(400, 300))
#         self.bias2 = nn.Parameter(torch.Tensor(1))
#
#
#     def forward(self, inputs):
#         text, aspect_text, position_tag = inputs[0], inputs[1], inputs[2]
#         '''Sentence reresentation '''
#         x = self.text_embed(text)
#         x_len = torch.sum(text != 0, dim=-1)
#         # x = [batch_size, sen len, embed dim]
#         aspect = self.text_embed(aspect_text) #(bs, 80, 300)
#         aspect_len = torch.sum(aspect_text != 0, dim=-1)
#         position = self.pos_embed(position_tag)
#         # position = [batch_size, sen len, pos dim(100)]
#         x = torch.cat((x, position), dim=-1)
#
#         conved1 = [F.tanh(conv(torch.transpose(x, 1, 2))).squeeze(-1) for conv in self.convs1]
#
#         # S = [torch.transpose(S1,1,2), torch.transpose(S3,1,2), torch.transpose(S5,1,2)]
#
#         S = torch.cat(conved1, dim=1) #(bs,300,80)
#
#         # alpha = F.softmax(F.tanh(torch.add(torch.bmm(torch.matmul(torch.unsqueeze(aspect_v, dim=1), self.weight2), S), self.bias2)), dim=1)
#         # pool = torch.bmm(alpha, torch.transpose(S, 1, 2)).squeeze(1)
#
#         x_r, _ = self.right_gru(x, x_len) # (bs, s, 400)
#         alpha_r = F.softmax(F.tanh(torch.add(torch.bmm(torch.matmul(x_r, self.weight2), torch.transpose(aspect, 1, 2)), self.bias2)),dim=1) #(bs, s, 80)
#         x_r_2 = torch.transpose(torch.bmm(torch.transpose(alpha_r,1,2), x_r),1,2) #(bs, 400, 80)
#         # x_r_2 = torch.bmm(torch.transpose(alpha_r,1,2), x_r) #(bs, 400, 80)
#         pool_r = F.max_pool1d(x_r_2, x_r_2.shape[2]).squeeze(2)
#         pool = F.avg_pool1d(S, S.shape[2]).squeeze(2) # (bs, 300)
#
#         pool = torch.cat((pool, pool_r), dim=-1)
#         out = F.tanh(self.fc1(pool))
#         out = self.fc2(out)
#
#         return out

'''# 完整体2'''
# class TextCNN(nn.Module):
#     '''CNN model'''
#     def __init__(self, embedding_matrix, opt):
#         super(TextCNN, self).__init__()
#         self.text_embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
#         self.pos_embed = nn.Embedding(opt.max_length, opt.position_dim)
#         self.right_gru = DynamicLSTM(opt.embed_dim+opt.position_dim, opt.hidden_dim, num_layers=1,
#                                     batch_first=True, bidirectional=True, rnn_type='GRU')
#         self.left_gru = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1,
#                                     batch_first=True, bidirectional=True, rnn_type='GRU')
#         self.weight_m = nn.Parameter(torch.Tensor(opt.hidden_dim*2, opt.hidden_dim*2))
#         self.bias_m = nn.Parameter(torch.Tensor(1))
#         self.weight_n = nn.Parameter(torch.Tensor(opt.hidden_dim*2, opt.hidden_dim*2))
#         self.bias_n = nn.Parameter(torch.Tensor(1))
#         self.fc1 = nn.Linear(700, 350)
#         self.fc2 = nn.Linear(350, opt.polarities_dim)
#
#         self.convs1 = nn.ModuleList([
#                                     nn.Conv1d(in_channels=opt.embed_dim + opt.position_dim,
#                                               out_channels=100,
#                                               kernel_size=fs,
#                                               padding=fs//2)
#                                     for fs in [1,1,1]
#                                     ])
#         self.convs3 = nn.ModuleList([
#                                     nn.Conv1d(in_channels=opt.embed_dim + opt.position_dim,
#                                               out_channels=100,
#                                               kernel_size=fs,
#                                               padding=fs//2)
#                                     for fs in [3,3,3]
#                                     ])
#         self.convs5 = nn.ModuleList([
#                                     nn.Conv1d(in_channels=opt.embed_dim + opt.position_dim,
#                                               out_channels=100,
#                                               kernel_size=fs,
#                                               padding=fs//2)
#                                     for fs in [5,5,5]
#                                     ])
#
#         self.weight1 = nn.Parameter(torch.Tensor(100, 100))
#         self.bias1 = nn.Parameter(torch.Tensor(1))
#         self.weight3 = nn.Parameter(torch.Tensor(100, 100))
#         self.bias3 = nn.Parameter(torch.Tensor(1))
#         self.weight5 = nn.Parameter(torch.Tensor(100, 100))
#         self.bias5 = nn.Parameter(torch.Tensor(1))
#         self.weight2 = nn.Parameter(torch.Tensor(400,300))
#         self.bias2 = nn.Parameter(torch.Tensor(1))
#
#
#     def forward(self, inputs):
#         text, aspect_text, position_tag = inputs[0], inputs[1], inputs[2]
#         '''Sentence reresentation '''
#         x = self.text_embed(text)
#         x_len = torch.sum(text != 0, dim=-1)
#         # x = [batch_size, sen len, embed dim]
#         aspect = self.text_embed(aspect_text) #(bs, 80, 300)
#         aspect_len = torch.sum(aspect_text != 0, dim=-1)
#         position = self.pos_embed(position_tag)
#         # position = [batch_size, sen len, pos dim(100)]
#         x = torch.cat((x, position), dim=-1)
#
#
#         aspect_v = aspect.sum(1) / aspect.size(1) # (bs, 300)
#
#
#         conved1 = [F.tanh(conv(torch.transpose(x, 1, 2))).squeeze(-1) for conv in self.convs1]
#         conved3 = [F.tanh(conv(torch.transpose(x, 1, 2))).squeeze(-1) for conv in self.convs3]
#         conved5 = [F.tanh(conv(torch.transpose(x, 1, 2))).squeeze(-1) for conv in self.convs5]
#         alpha1 = F.softmax(F.tanh(torch.add(torch.bmm(torch.matmul(torch.transpose(conved1[0], 1, 2), self.weight1), conved1[1]), self.bias1)), dim=1) #(bs,80,80)
#
#         S1 = torch.bmm(alpha1, torch.transpose(conved1[2], 1, 2))#(bs,80,100)
#         alpha3 = F.softmax(F.tanh(torch.add(torch.bmm(torch.matmul(torch.transpose(conved3[0], 1, 2), self.weight3), conved3[1]), self.bias3)), dim=1) #(bs,80,80)
#         S3 = torch.bmm(alpha3, torch.transpose(conved3[2], 1, 2))#(bs,80,100)
#         alpha5 = F.softmax(F.tanh(torch.add(torch.bmm(torch.matmul(torch.transpose(conved5[0], 1, 2), self.weight5), conved5[1]), self.bias5)), dim=1) #(bs,80,80)
#         S5 = torch.bmm(alpha5, torch.transpose(conved5[2], 1, 2))#(bs,80,100)
#
#
#
#         # S = [torch.transpose(S1,1,2), torch.transpose(S3,1,2), torch.transpose(S5,1,2)]
#         S = torch.transpose(torch.cat((S1,S3,S5), dim=2), 1, 2) #(bs,300,80)
#
#         # alpha = F.softmax(F.tanh(torch.add(torch.bmm(torch.matmul(torch.unsqueeze(aspect_v, dim=1), self.weight2), S), self.bias2)), dim=1)
#         # pool = torch.bmm(alpha, torch.transpose(S, 1, 2)).squeeze(1)
#
#         x_r, _ = self.right_gru(x, x_len) # (bs, s, 400)
#         # alpha_r = F.softmax(F.tanh(torch.add(torch.bmm(torch.matmul(x_r, self.weight2), torch.transpose(aspect, 1, 2)), self.bias2)), dim=1) #(bs, 80,s)
#         alpha_r = F.softmax(F.tanh(torch.add(torch.bmm(torch.matmul(aspect, self.weight2.transpose(1,0)), torch.transpose(x_r, 1, 2)), self.bias2)), dim=1)
#         print(alpha_r.shape)
#
#         with open('alpha', 'a') as f:
#             # f.write(str(self.weight1.squeeze(0).sum(dim=0).cpu().numpy())+'\n')
#             # f.write(str(self.weight1.squeeze(0).sum(dim=1).cpu().numpy())+'\n')
#             f.write('------------\n')
#             f.write('-------------------------------------------------\n')
#             alpha_pool = alpha_r.mean(dim=2)
#             f.write(str(alpha_pool.cpu().numpy())+'\n')
#             f.write(str(alpha_r)+'\n')
#             f.write(str(alpha_r.squeeze(0).sum(dim=1).cpu().numpy())+'\n')
#             # f.write(str(alpha_r.squeeze(0).sum(dim=2).cpu().numpy())+'\n')
#             f.write('>>>>>>>>>>>>\n')
#
#         x_r_2 = torch.transpose(torch.bmm(alpha_r, x_r),1,2) #(bs, 400, 80)
#         # x_r_2 = torch.bmm(torch.transpose(alpha_r,1,2), x_r) #(bs, 400, 80)
#         pool_r = F.max_pool1d(x_r_2, x_r_2.shape[2]).squeeze(2)
#         pool = F.avg_pool1d(S, S.shape[2]).squeeze(2) # (bs, 300)
#
#         pool = torch.cat((pool, pool_r), dim=-1)
#         out = F.tanh(self.fc1(pool))
#         out = self.fc2(out)
#
#         # conved1 = [F.tanh(conv(torch.transpose(x, 1, 2))).squeeze(-1) for conv in self.convs1] # (bs, 100, 80(81))
#         # alpha = F.softmax(F.tanh(torch.add(torch.bmm(torch.matmul(torch.transpose(conved1[0], 1, 2), self.weight1), conved1[1]), self.bias1)), dim=1) #(bs,80,80)
#         # S = torch.bmm(alpha, torch.transpose(conved1[2], 1, 2))#(bs,80,400)
#         # S_pool = torch.unsqueeze(torch.div(torch.sum(S, dim=1), x_len.float().view(x_len.size(0), 1)), dim=1)
#         # h_t = [F.tanh(conv(torch.transpose(aspect, 1, 2))) for conv in self.convs1t]
#         # # print(h_t[0].shape)
#         # h_t = torch.transpose(torch.cat(h_t, dim=1),1,2)
#         #
#         # alpha2 = F.softmax(F.tanh(torch.add(torch.bmm(torch.matmul(h_t, self.weight_m), torch.transpose(S, 1, 2)), self.bias_m)), dim=1)
#         # S_x = torch.bmm(alpha2, S)
#         #
#         # gamma = F.softmax(F.tanh(torch.add(torch.bmm(torch.matmul(S_pool, self.weight_n), torch.transpose(h_t, 1, 2)), self.bias_n)), dim=1)
#         # h_r = torch.squeeze(torch.bmm(gamma, S_x), dim=1)
#         # #
#         # # pool = F.max_pool1d(S, S.shape[2]).squeeze(2)
#         # # cat = torch.cat((pool, pool_t), dim=-1)
#         # out = F.tanh(self.fc1(h_r))
#         # out = self.fc2(out)
#
#         '''self-attention only x and pos [1,3,5]---- 0.80625'''
#         # conved1 = [F.tanh(conv(torch.transpose(x, 1, 2))).squeeze(-1) for conv in self.convs1] # (bs, 100, 80(81))
#         # alpha = F.softmax(F.tanh(torch.add(torch.bmm(torch.matmul(torch.transpose(conved1[0], 1, 2), self.weight1), conved1[1]), self.bias1)), dim=1)
#         # S = torch.transpose(torch.bmm(alpha, torch.transpose(conved1[2], 1, 2)), 1, 2)
#         # pool = F.max_pool1d(S, S.shape[2]).squeeze(2)
#         # out = F.tanh(self.fc1(pool))
#         # out = self.fc2(out)
#
#         ''' aspect attention x --- 0.7946'''
#         # conved = [F.tanh(conv(torch.transpose(x, 1, 2))).squeeze(-1) for conv in self.convs1] # (bs, 100, 80(81))
#         # # print(conved[0].shape)
#         # conved2 = [F.tanh(conv(torch.transpose(aspect, 1, 2))).squeeze(-1) for conv in self.convs2] # (bs, 100, 80)
#         # # print(conved2[0].shape)
#         # alpha = [F.softmax(F.tanh(torch.add(torch.bmm(torch.matmul(torch.transpose(i, 1, 2), self.weight1), j), self.bias1)), dim=1)  for j, i in zip(conved, conved2)] # (bs, 80, 80)
#         # # print(alpha[0].shape)
#         # s = [torch.bmm(a, torch.transpose(conv, 1, 2)) for a, conv in zip(alpha, conved)] # (bs, 80, 100)\
#         # s = [torch.transpose(i, 1, 2) for i in s]
#         # # print(s[0].shape)
#         # pool = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in s] # (bs, 100)
#         # # print(pool[0].shape)
#         # out = torch.cat(pool, 1)
#         # out = F.tanh(self.fc1(out)) # (bs, 300)
#         # out = self.fc2(out) # (bs, 150)
#
#         '''only cnn [1,3,5] 0.7982'''
#         # pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved1]# (bs, 100)
#         # cat = torch.cat(pooled, 1)
#         # out = F.tanh(self.fc1(cat))
#         # out = self.fc2(out)
#         return out
