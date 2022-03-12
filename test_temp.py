import torch
import nltk
import string
from tqdm import tqdm

from file_utils import read_file_in_dir
from dataset_factory import get_datasets
from model_factory import get_model
from caption_utils import bleu1, bleu4


def sequences_to_words(captions):
    words = []
    for i in captions:
        caption = nltk.word_tokenize(i)
        tmp = []
        for word in caption:
            if word not in string.punctuation:
                tmp.append(word.lower())
        words.append(tmp)
    return words


def vec_to_words(captions):
    words = []
    for i in captions:
        if i > 3:
            word = vocab.idx2word[i.item()].lower()
            if word not in string.punctuation:
                words.append(vocab.idx2word[i.item()].lower())
        if i == 2:
            break
    return words


def calc_bleu1(captions, generate_captions):
    reference_captions = []
    predicted_captions = []
    for vec in captions:
        reference_captions.append(sequences_to_words(vec))
    for vec in generate_captions:
        predicted_captions.append(vec_to_words(vec))
    bleu_value = 0
    for i in range(len(reference_captions)):
        bleu_value += bleu1(reference_captions[i], predicted_captions[i])
    bleu_value /= len(reference_captions)
    return bleu_value


def calc_bleu4(captions, generate_captions):
    reference_captions = []
    predicted_captions = []
    for vec in captions:
        reference_captions.append(sequences_to_words(vec))
    for vec in generate_captions:
        predicted_captions.append(vec_to_words(vec))
    bleu_value = 0
    for i in range(len(reference_captions)):
        bleu_value += bleu4(reference_captions[i], predicted_captions[i])

    bleu_value /= len(reference_captions)
    return bleu_value


if __name__ == "__main__":

    config_data = read_file_in_dir('./', 'default.json')
    coco_test, vocab, train_loader, val_loader, test_loader = get_datasets(config_data)

    encoder, decoder = get_model(config_data, vocab)

    # TODO: Change the
    state_dict = torch.load(r'experiment_data/LSTM_larger3/checkpoint_model.pt')
    encoder.load_state_dict(state_dict['encoder_state'])
    decoder.load_state_dict(state_dict['decoder_state'])

    temperatures = [0.1, 0.2, 0.7, 1, 1.5, 2]

    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        if torch.cuda.is_available():
            encoder = encoder.cuda()
            decoder = decoder.cuda()

        # Bleu-1 and 4 for deterministic
        deterministic_bleu1 = 0
        deterministic_bleu4 = 0
        # Bleu-1 and 4 for stochastic with temperature=0.1
        stochastic_bleu1_01 = 0
        stochastic_bleu4_01 = 0
        # Bleu-1 and 4 for stochastic with temperature=0.2
        stochastic_bleu1_02 = 0
        stochastic_bleu4_02 = 0
        # Bleu-1 and 4 for stochastic with temperature=0.7
        stochastic_bleu1_07 = 0
        stochastic_bleu4_07 = 0
        # Bleu-1 and 4 for stochastic with temperature=1
        stochastic_bleu1_1 = 0
        stochastic_bleu4_1 = 0
        # Bleu-1 and 4 for stochastic with temperature=1.5
        stochastic_bleu1_15 = 0
        stochastic_bleu4_15 = 0
        # Bleu-1 and 4 for stochastic with temperature=2
        stochastic_bleu1_2 = 0
        stochastic_bleu4_2 = 0

        for _, (images, captions, img_ids, length) in enumerate(tqdm(test_loader)):
            captions = captions.cuda()
            ground_captions = [[i['caption'] for i in coco_test.imgToAnns[idx]] for idx in img_ids]
            encoder_output = encoder(images.cuda())
            deterministic_captions = decoder.generate(encoder_output, max_length=config_data['generation']['max_length'],
                                                      stochastic=False, temp=config_data['generation']['temperature'])
            stochastic_01_captions = decoder.generate(encoder_output, max_length=config_data['generation']['max_length'],
                                                      stochastic=True, temp=0.1)
            stochastic_02_captions = decoder.generate(encoder_output, max_length=config_data['generation']['max_length'],
                                                      stochastic=True, temp=0.2)
            stochastic_07_captions = decoder.generate(encoder_output, max_length=config_data['generation']['max_length'],
                                                      stochastic=True, temp=0.7)
            stochastic_1_captions = decoder.generate(encoder_output, max_length=config_data['generation']['max_length'],
                                                     stochastic=True, temp=1)
            stochastic_15_captions = decoder.generate(encoder_output, max_length=config_data['generation']['max_length'],
                                                      stochastic=True, temp=1.5)
            stochastic_2_captions = decoder.generate(encoder_output, max_length=config_data['generation']['max_length'],
                                                     stochastic=True, temp=2)
            deterministic_bleu1 += calc_bleu1(ground_captions, deterministic_captions)
            deterministic_bleu4 += calc_bleu4(ground_captions, deterministic_captions)

            stochastic_bleu1_01 += calc_bleu1(ground_captions, stochastic_01_captions)
            stochastic_bleu4_01 += calc_bleu4(ground_captions, stochastic_01_captions)

            stochastic_bleu1_02 += calc_bleu1(ground_captions, stochastic_02_captions)
            stochastic_bleu4_02 += calc_bleu4(ground_captions, stochastic_02_captions)

            stochastic_bleu1_07 += calc_bleu1(ground_captions, stochastic_07_captions)
            stochastic_bleu4_07 += calc_bleu4(ground_captions, stochastic_07_captions)

            stochastic_bleu1_1 += calc_bleu1(ground_captions, stochastic_1_captions)
            stochastic_bleu4_1 += calc_bleu4(ground_captions, stochastic_1_captions)

            stochastic_bleu1_15 += calc_bleu1(ground_captions, stochastic_15_captions)
            stochastic_bleu4_15 += calc_bleu4(ground_captions, stochastic_15_captions)

            stochastic_bleu1_2 += calc_bleu1(ground_captions, stochastic_2_captions)
            stochastic_bleu4_2 += calc_bleu4(ground_captions, stochastic_2_captions)

        result = [
            deterministic_bleu1 / len(test_loader),
            deterministic_bleu4 / len(test_loader),

            stochastic_bleu1_01 / len(test_loader),
            stochastic_bleu4_01 / len(test_loader),

            stochastic_bleu1_02 / len(test_loader),
            stochastic_bleu4_02 / len(test_loader),

            stochastic_bleu1_07 / len(test_loader),
            stochastic_bleu4_07 / len(test_loader),

            stochastic_bleu1_1 / len(test_loader),
            stochastic_bleu4_1 / len(test_loader),

            stochastic_bleu1_15 / len(test_loader),
            stochastic_bleu4_15 / len(test_loader),

            stochastic_bleu1_2 / len(test_loader),
            stochastic_bleu4_2 / len(test_loader),
        ]

        print(result)
