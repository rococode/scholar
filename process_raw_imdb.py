import os

import nltk
from tqdm import tqdm

nltk.download('punkt')

TRAIN_DIR = "./data/imdb/imdb_train"
TEST_DIR = "./data/imdb/imdb_test"

OUT_TRAIN_DIR = "./data/imdb/out_imdb_train"
OUT_TEST_DIR = "./data/imdb/out_imdb_test"

if not os.path.exists(OUT_TRAIN_DIR):
    os.makedirs(OUT_TRAIN_DIR)
if not os.path.exists(OUT_TEST_DIR):
    os.makedirs(OUT_TEST_DIR)

TRAIN_FILE = 'train.jsonlist'
TEST_FILE = 'test.jsonlist'


def tokenize_and_save(file_path, name, file_dest, type, sentiment):
    doc_id, rating = name.split('_')
    # print("file dest", file_dest)
    with open(file_path, "r") as f:
        lines = f.readlines()
    sentences = []
    for l in lines:
        l = l.replace('<br />', ' ')
        l = l.strip()
        if len(l) == 0:
            continue
        for sentence in nltk.sent_tokenize(l):
            sentences.append(sentence + '\n')
    if not os.path.exists(file_dest):
        os.makedirs(file_dest)
    with open(file_dest + os.sep + name, "w") as out:
        out.writelines(sentences)
    # doc = {'id': type + '_' + str(doc_id), 'text': sentences, 'sentiment': sentiment, 'orig': file_path, 'rating': rating}
    # print(doc)


def process(dir, dest, type):
    for dir_name, subdir_list, file_list in os.walk(dir):
        # print(dir_name, file_list[0] if len(file_list) > 0 else '')
        for file_name in tqdm(file_list):
            sentiment = dir_name.split(os.sep)[-1]
            # print("sent", sentiment)
            path = dir_name + os.sep + file_name
            # print(path)
            tokenize_and_save(path, file_name, dest + os.sep + sentiment, type, sentiment)
            # break

process(TRAIN_DIR, OUT_TRAIN_DIR, "train")
process(TEST_DIR, OUT_TEST_DIR, "test")
