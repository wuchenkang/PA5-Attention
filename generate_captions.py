import torch
import nltk
import string
from tqdm import tqdm

from file_utils import read_file_in_dir
from dataset_factory import get_datasets
from model_factory import get_model
from caption_utils import bleu1
from visual import draw_caption


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


def convert_to_sentence(word_indices):
    result = []
    for word_idx in word_indices:
        word = vocab.idx2word[word_idx].lower()
        if word == "<end>":
            break
        elif word in string.punctuation or word == "<start>":
            continue
        else:
            result.append(word)
    return result


if __name__ == '__main__':

    # TODO: Change the directory of model configuration file
    config_data = read_file_in_dir('./', 'default.json')
    coco_test, vocab, train_loader, val_loader, test_loader = get_datasets(config_data)

    encoder, decoder = get_model(config_data, vocab)

    # TODO: Change the directory of model weight file
    state_dict = torch.load(r'experiment_data/LSTM_larger3/checkpoint_model.pt')
    encoder.load_state_dict(state_dict['encoder_state'])
    decoder.load_state_dict(state_dict['decoder_state'])

    encoder.eval()
    decoder.eval()

    if torch.cuda.is_available():
        encoder = encoder.cuda()
        decoder = decoder.cuda()

    with torch.no_grad():
        img_captions = []
        for _, (images, captions, img_ids, length) in enumerate(tqdm(test_loader)):
            captions = captions.cuda()
            ground_captions = [[i['caption'] for i in coco_test.imgToAnns[idx]] for idx in img_ids]
            encoder_output = encoder(images.cuda())
            generate_captions = decoder.generate(encoder_output, max_length=config_data['generation']['max_length'],
                                                 stochastic=not config_data['generation']['deterministic'],
                                                 temp=config_data['generation']['temperature'])

            for index, img_id in enumerate(img_ids):
                generated_caption_indices = generate_captions[index].cpu().detach().numpy()
                generated_caption = convert_to_sentence(generated_caption_indices)
                captions_bleu = calc_bleu1([ground_captions[index]], [generated_caption_indices])
                img_captions.append((img_id, captions_bleu, ground_captions[index], generated_caption))

    img_captions.sort(key=lambda tup: tup[1], reverse=True)

    top_10_captions = img_captions[:10]

    img_captions = img_captions[::-1]
    bottom_10_captions = []
    bottom_10_ids = []

    for img_caption in img_captions:
        if img_caption[0] in bottom_10_ids:
            continue
        bottom_10_captions.append(img_caption)
        bottom_10_ids.append(img_caption[0])
        if len(bottom_10_ids) == 10:
            break

    print(top_10_captions)
    print(bottom_10_captions)

    # TODO: Change the directory of generated images
    for index, tup in enumerate(top_10_captions):
        draw_caption(tup, f"good_caption_{index + 1}.jpg")

    for index, tup in enumerate(bottom_10_captions):
        draw_caption(tup, f"bad_caption_{index + 1}.jpg")
