import random

CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
CLASSES = len(CHARS)
TOTAL_CHARS = 6

IMG_WIDTH = 160
IMG_HEIGHT = 60
IMG_DEPTH = 3

EPOCHS = 60
MINI_BATCH_SIZE = 100

TRAIN_DATA_TO_LOAD = 25000
TEST_DATA_TO_LOAD = 9000
VALIDATION_DARA_TO_LOAD = 8000


def randomize(arr1, arr2):
    combined = list(zip(arr1, arr2))
    random.shuffle(combined)
    return zip(*combined)


def vec_to_char(v):
    return "".join(CHARS[j % CLASSES] for j in range(len(v)) if v[j] >= 0.5)

def vec_to_captcha(v):
    return "".join(CHARS[i] for i in v)