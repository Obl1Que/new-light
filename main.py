import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import cv2


def rgb_unique(img):
    return np.unique(img.reshape(-1, img.shape[2]), axis=0)


def quantize_image(img, pallete):
    img_pil = Image.fromarray(img)
    conv = img_pil.quantize(colors=29, palette=pallete, dither=0)
    conv = conv.convert('RGB')
    img_label = np.array(conv)
    return img_label


classes_colors = {'Truck': [255, 0, 0],
                  'Tank': [255, 255, 0],
                  'BTR': [0, 255, 0],
                  'Car': [0, 255, 255],
                  'Minibus': [0, 0, 255],
                  'Human': [255, 0, 255],
                  'Building': [255, 255, 255],
                  'Road': [127, 0, 0],
                  'Trees': [0, 127, 0],
                  'Grass': [0, 127, 127],
                  'Concrete': [0, 0, 127],
                  'Water': [127, 127, 127],
                  'DMost': [99, 161, 140],
                  'OporaM': [167, 100, 89],
                  'Ferma': [189, 189, 189],
                  'Cloud': [0, 200, 150],
                  'Sky': [160, 160, 160],
                  'Tube': [255, 100, 50],
                  'Tower': [90, 150, 110],
                  'Lamppost': [150, 100, 50],
                  'Bilboard': [255, 234, 128],
                  'Fence': [160, 160, 60],
                  'Animal': [61, 39, 60],
                  'Bridge': [120, 160, 140],
                  'Ship': [128, 26, 34],
                  'Pillar': [150, 100, 90],
                  'Bridge Auto': [120, 160, 140],
                  'Truss': [188, 188, 188],
                  'Other': [200, 150, 200],
                  'Window': [0, 150, 255],
                  'Other2': [200, 150, 200],
                  'Other3': [200, 150, 200]
                  }
palette_quant = []
j = 0
for k, v in classes_colors.items():
    palette_quant.extend(v)
    j += 1
img_p = Image.new("P", (1, 1))
img_p.putpalette(palette_quant + [0] * (768 - len(palette_quant)))

in_img = np.array(Image.open(r'C:\Users\sdezh\PycharmProjects\new-light\NEWW\IN\ColorImages\Скриншот 18-02-2023 090135.png'))
in_sem = np.array(Image.open(r'C:\Users\sdezh\PycharmProjects\new-light\NEWW\IN\Class\Скриншот 18-02-2023 090135.png'))
like_img = np.array(Image.open(r'C:\Users\sdezh\PycharmProjects\new-light\NEWW\Target\ColorImages\Скриншот 18-02-2023 092521.png'))
like_sem = np.array(Image.open(r'C:\Users\sdezh\PycharmProjects\new-light\NEWW\Target\Class\Скриншот 18-02-2023 092521.png'))

colors = rgb_unique(in_sem)
c = 0

for col in colors:
    src_mask = np.all(in_sem == col, axis=-1).astype('bool')
    plt.imsave(f'{c}.jpg', src_mask)
    c += 1
