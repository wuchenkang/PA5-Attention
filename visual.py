import matplotlib.pyplot as plt
from PIL import Image


def draw_caption(tup, path):
    img_id, _, captions, generated_captions = tup
    generated_captions[0] = generated_captions[0][0].upper()
    generated_captions = " ".join(generated_captions) + "."

    img_path = r"data/images/test/COCO_val2014_" + str(img_id).zfill(12) + ".jpg"
    img = Image.open(img_path)
    width, height = img.size
    plt.figure(figsize=(15, 27))
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    start = height + 14
    plt.text(0, start, "Reference Captions:", fontsize=20)
    start += 20
    for index, caption in enumerate(captions):
        plt.text(0, start, caption, fontsize=20)
        start += 20
    plt.text(0, start, "Generated Caption:", fontsize=20)
    start += 20
    plt.text(0, start, generated_captions, fontsize=20)
    start += 20
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
