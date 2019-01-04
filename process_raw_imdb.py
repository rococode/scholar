"""
This is not part of SCHOLAR, it's actually for prepping the downloaded
IMDB dataset for the open-sesame code. I just ended up writing it here
since it's closer to the directory that the imdb set is downloaded into.
"""

import os
import json

import nltk
from tqdm import tqdm
import file_handling as fh

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
    print(lines)
    stopword_list = fh.read_text(os.path.join('stopwords', 'snowball_stopwords.txt'))
    stopword_set = {s.strip() for s in stopword_list}
    print(stopword_list)
    for l in lines:
        from preprocess_data import tokenize
        for sentence in nltk.sent_tokenize(l):
            print(sentence)
            tokens, _counts = tokenize(sentence, stopwords=stopword_set)
            print(tokens)
            sentences.append(json.dumps(tokens) + '\n')
        import sys
        sys.exit(0)
        l = l.strip()
        if len(l) == 0:
            continue
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
