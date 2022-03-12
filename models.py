#from turtle import forward
#from black import out
import numpy as np
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
        self.mapping = nn.Sequential(
            nn.Conv2d(1, 3, (3, 3), stride=(2, 2)),
            nn.ReLU(),
            nn.BatchNorm2d(3),
            nn.MaxPool2d(4, 4),
        )

    def forward(self, x):
        x = x.cpu()
        batch_input = []
        for image in x:
            inputs = self.feature_extractor(image, return_tensors="pt")
            values = inputs["pixel_values"]
            if torch.cuda.is_available():
                values = values.cuda()
            batch_input.append(values)
        inputs = {"pixel_values": torch.stack(batch_input, dim=1).squeeze(0)}

        with torch.no_grad():
            outputs = self.model(**inputs).last_hidden_state
        outputs = self.mapping(outputs.unsqueeze(1))
        outputs = outputs.reshape(outputs.size(0), -1)

        return outputs


class Attention(nn.Module):
    """ Applies attention mechanism on the `context` using the `query`.

    **Thank you** to IBM for their initial implementation of :class:`Attention`. Here is
    their `License
    <https://github.com/IBM/pytorch-seq2seq/blob/master/LICENSE>`__.

    Args:
        dimensions (int): Dimensionality of the query and context.
        attention_type (str, optional): How to compute the attention score:

            * dot: :math:`score(H_j,q) = H_j^T q`
            * general: :math:`score(H_j, q) = H_j^T W_a q`

    Example:

         >>> attention = Attention(256)
         >>> query = torch.randn(5, 1, 256)
         >>> context = torch.randn(5, 5, 256)
         >>> output, weights = attention(query, context)
         >>> output.size()
         torch.Size([5, 1, 256])
         >>> weights.size()
         torch.Size([5, 1, 5])
    """

    def __init__(self, dimensions, attention_type='general'):
        super(Attention, self).__init__()

        if attention_type not in ['dot', 'general']:
            raise ValueError('Invalid attention type selected.')

        self.attention_type = attention_type
        if self.attention_type == 'general':
            self.linear_in = nn.Linear(dimensions, dimensions, bias=False)

        self.linear_out = nn.Linear(dimensions * 2, dimensions, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

    def forward(self, query, context):
        """
        Args:
            query (:class:`torch.FloatTensor` [batch size, output length, dimensions]): Sequence of
                queries to query the context.
            context (:class:`torch.FloatTensor` [batch size, query length, dimensions]): Data
                overwhich to apply the attention mechanism.

        Returns:
            :class:`tuple` with `output` and `weights`:
            * **output** (:class:`torch.LongTensor` [batch size, output length, dimensions]):
              Tensor containing the attended features.
            * **weights** (:class:`torch.FloatTensor` [batch size, output length, query length]):
              Tensor containing attention weights.
        """
        batch_size, output_len, dimensions = query.size()
        query_len = context.size(1)

        if self.attention_type == "general":
            query = query.reshape(batch_size * output_len, dimensions)
            query = self.linear_in(query)
            query = query.reshape(batch_size, output_len, dimensions)

        # TODO: Include mask on PADDING_INDEX?

        # (batch_size, output_len, dimensions) * (batch_size, query_len, dimensions) ->
        # (batch_size, output_len, query_len)
        attention_scores = torch.bmm(query, context.transpose(1, 2).contiguous())

        # Compute weights across every context sequence
        attention_scores = attention_scores.view(batch_size * output_len, query_len)
        attention_weights = self.softmax(attention_scores)
        attention_weights = attention_weights.view(batch_size, output_len, query_len)

        # (batch_size, output_len, query_len) * (batch_size, query_len, dimensions) ->
        # (batch_size, output_len, dimensions)
        mix = torch.bmm(attention_weights, context)

        # concat -> (batch_size * output_len, 2*dimensions)
        combined = torch.cat((mix, query), dim=2)
        combined = combined.view(batch_size * output_len, 2 * dimensions)

        # Apply linear_out on every 2nd dimension of concat
        # output -> (batch_size, output_len, dimensions)
        output = self.linear_out(combined).view(batch_size, output_len, dimensions)
        output = self.tanh(output)

        return output, attention_weights


class LstmDecoder(nn.Module):
    def __init__(self, encoder_dim, embedding_dim, vocab_size, decoder_dim, decoder_depth, attention=True, lstm_flag = True):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.embedding_dim = embedding_dim
        self.decoder_dim = decoder_dim
        self.decoder_depth = decoder_depth
        
        self.fc1 = nn.Linear(encoder_dim, embedding_dim)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        self.lstm_flag = lstm_flag
        if lstm_flag:
            self.lstm = nn.LSTM(embedding_dim, decoder_dim, decoder_depth, batch_first=True)
        else:
            self.rnn = nn.RNN(embedding_dim, decoder_dim, decoder_depth, nonlinearity='relu', batch_first=True)
            
        self.fc2 = nn.Linear(decoder_dim, vocab_size)
        self.softmax = nn.Softmax(dim=2)
        self.bn = nn.BatchNorm1d(embedding_dim, momentum=0.01)
        
        self.dropout = nn.Dropout(0.2)
        
        self.attention = Attention(decoder_dim, attention_type='dot')
        
        self.decoder_cell = nn.LSTM(decoder_dim, decoder_dim, decoder_depth, batch_first=True)
        

    def init_hidden_states(self, batch_size, device):
        hidden_state = torch.zeros((self.decoder_depth, batch_size, self.decoder_dim)).to(device)
        cell_state = torch.zeros((self.decoder_depth, batch_size, self.decoder_dim)).to(device)
        return hidden_state, cell_state

    def forward(self, x, y, lengths):
        x = self.bn( self.fc1(x) )
        
        #x = self.dropout(x)
        
        feature = x.view((x.shape[0], 1, x.shape[1]))
        
        y = self.embedding(y)
        
        x = torch.cat((feature, y), dim=1)
        x = pack_padded_sequence(x, lengths, batch_first = True)
        h0, c0 = self.init_hidden_states(y.shape[0], y.device)
        
        if self.lstm_flag:
            output, _ = self.lstm(x, (h0.detach(), c0.detach() ))
        else:
            output, _ = self.rnn(x, h0.detach())
            
        output, _ = pad_packed_sequence(output, batch_first=True)
        
        #output is [batch, L, decoder_dim=512]

        attn = torch.zeros_like(output)

        for i in range( lengths[0] ):
            temp, _ = self.attention(output[:,i,:].unsqueeze(1), output[:,:(i+1),:])
            temp = temp.squeeze(1)
            attn[:,i,:] = temp   
        
        h0, c0 = self.init_hidden_states(y.shape[0], y.device)
        attn = pack_padded_sequence(attn, [lengths[0] for i in range(len(lengths)) ], batch_first = True)
        output, _ = self.decoder_cell(attn, (h0.detach(), c0.detach()))
        output, _ = pad_packed_sequence(output, batch_first=True)  
        
        predictions = self.fc2(output)
        
        return predictions

    def generate(self, x, max_length=100, stochastic = False, temp=0.1):
        # batch * image
        x = self.bn( self.fc1(x) )
        x = x.view((x.shape[0], 1, x.shape[1]))

        pred = torch.zeros((x.size(0), max_length), dtype=torch.long).cuda()
        h, c = self.init_hidden_states(x.shape[0], x.device)
        h = h.detach()
        c = c.detach()
        
        h_decoder, c_decoder = self.init_hidden_states(x.shape[0], x.device)
        h_decoder = h_decoder.detach()
        c_decoder = c_decoder.detach()
        
        hidden_all = torch.zeros( (x.size(0), max_length, self.decoder_dim) ).to(x.device)
        
        for t in range(max_length):
            # print(t)
            if self.lstm_flag:
                output, (h, c) = self.lstm(x, (h, c))
            else:
                output, h = self.rnn(x, h)
            
            #output = [B, 1, dim]
            hidden_all[:,t,:] = output.squeeze(1)
            output, _ = self.attention(output, hidden_all[:,:(t+1),:])
            
            output, (h_decoder, c_decoder) = self.decoder_cell( output , (h_decoder, c_decoder) )
            
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
