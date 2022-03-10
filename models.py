from turtle import forward
from black import out
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet50
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import ViTFeatureExtractor, ViTModel


class ResnetEncoder(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        model = resnet50(pretrained = True)
        for param in model.parameters():
            param.requires_grad = False
        self.feature_map = nn.Sequential( *list(model.children())[:-1] )

        # self.embedding = nn.Linear(model.fc.in_features, embedding_dim)
        

    def forward(self, x):
        x = self.feature_map(x)
        x = x.reshape(x.size(0), -1)
        # x = self.embedding(x)
        return x


class VitEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
        self.model = ViTModel.from_pretrained('google/vit-base-patch16-224')

    def forward(self, x):
        inputs = self.feature_extractor(x, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs).last_hidden_state
        return outputs


class Attention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query, keys, values):
        """
        [B, S, L]
        """
        query = torch.unsqueeze(1) # [B, 1, L]
        keys = keys.transpose((1, 2)) # [B, L, S]
        alpha = torch.bmm(query, keys) # [B, 1, S]
        alpha = F.softmax(alpha, dim=2) # [B, 1, S]
        result = torch.bmm(alpha, values).squeeze(1) # [B, L]
        return result


class LstmDecoder(nn.Module):
    def __init__(self, encoder_dim, embedding_dim, vocab_size, decoder_dim, decoder_depth, attention=False):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.embedding_dim = embedding_dim
        self.decoder_dim = decoder_dim
        self.decoder_depth = decoder_depth
        
        self.fc1 = nn.Linear(encoder_dim, embedding_dim)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, decoder_dim, decoder_depth, batch_first=True)
        self.attention = Attention()
        self.fc2 = nn.Linear(decoder_dim, vocab_size)
        self.softmax = nn.Softmax(dim=2)

        self.bn = nn.BatchNorm1d(embedding_dim, momentum=0.01)

    def init_hidden_states(self, batch_size, device):
        hidden_state = torch.zeros((self.decoder_depth, batch_size, self.decoder_dim)).to(device)
        cell_state = torch.zeros((self.decoder_depth, batch_size, self.decoder_dim)).to(device)
        return hidden_state, cell_state

    def forward(self, x, y, lengths):
        x = self.bn( self.fc1(x) )
        x = x.view((x.shape[0], 1, x.shape[1]))
        y = self.embedding(y)
        x = torch.cat((x, y), dim=1)
        x = pack_padded_sequence(x, lengths, batch_first = True)
        h0, c0 = self.init_hidden_states(y.shape[0], y.device)
        output, _ = self.lstm(x, (h0.detach(), c0.detach()))
        output, _ = pad_packed_sequence(output, batch_first=True)
        attn = torch.zeros_like(output)
        if self.attention:
            length = lengths[0]
            for t in range(length):
                masked_output = output.copy()
                masked_output[t:] = 0
                attn[:, t, :] = self.attention(output[:, t, :], masked_output, masked_output)
        predictions = self.fc2(attn)
        # batch_size * max_len * vocab_size
        return predictions

    def generate(self, x, max_length=100, stochastic = False, temp=0.1):
        # batch * image
        x = self.bn( self.fc1(x) )
        x = x.view((x.shape[0], 1, x.shape[1]))

        pred = torch.zeros((x.size(0), max_length), dtype=torch.long).cuda()
        h, c = self.init_hidden_states(x.shape[0], x.device)
        h = h.detach()
        c = c.detach()
        for t in range(max_length):
            # print(t)
            output, (h, c) = self.lstm(x, (h, c)) # output dimension?
            output = self.fc2(output)
            #print(output.size())
            if stochastic:
                output = self.softmax(output/temp).reshape(output.size(0),-1)
                # batch_size * vocab_size
                pred[:,t] = torch.multinomial(output.data, 1).view(-1)
            else:
                #deterministic
                pred[:,t] = torch.argmax(output, dim=2).view(-1)

            x = self.embedding(pred[:, t])
            x = x.view((x.shape[0], 1, x.shape[1]))
        
        return pred

