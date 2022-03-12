################################################################################
# CSE 253: Programming Assignment 4
# Code snippet by Ajit Kumar, Savyasachi
# Fall 2020
################################################################################
from models import ResnetEncoder, VitEncoder, LstmDecoder


# Build and return the model here based on the configuration.
def get_model(config_data, vocab):
    hidden_size = config_data['model']['hidden_size']
    embedding_size = config_data['model']['embedding_size']
    encoder_type = config_data['model']['encoder_type']
    use_attention = config_data['model']['decoder_attention']

    encoder_dim = config_data['model']['encoder_dim']
    vocab_size = len(vocab)
    decoder_depth = config_data['model']['decoder_depth']
    # You may add more parameters if you want
    if encoder_type == "Resnet":
        print("Encoder using ResNet")
        encoder = ResnetEncoder(embedding_size)
    elif encoder_type == "ViT":
        print("Encoder using Visual Transformer")
        encoder = VitEncoder()
    else:
        raise ValueError(f"{encoder_type} encoder not supported, please choose from ['Resnet', 'ViT']")

    decoder = LstmDecoder(encoder_dim, embedding_size, vocab_size, hidden_size)

    return encoder, decoder
