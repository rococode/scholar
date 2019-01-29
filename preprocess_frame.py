import os
import re
import string
import sys
from collections import Counter
from optparse import OptionParser

import numpy as np
import pandas as pd
from scipy import sparse
from tqdm import tqdm

import file_handling as fh

"""
Convert a dataset into the required format (as well as formats required by other tools).
Input format is one line per item.
Each line should be a json object.
At a minimum, each json object should have a "text" field, with the document text.
Any other field can be used as a label (specified with the --label option).
If training and test data are to be processed separately, the same input directory should be used
Run "python preprocess_data -h" for more options.
If an 'id' field is provided, this will be used as an identifier in the dataframes, otherwise index will be used 
"""

# compile some regexes
punct_chars = list(set(string.punctuation) - set("'"))
punct_chars.sort()
punctuation = ''.join(punct_chars)
replace = re.compile('[%s]' % re.escape(punctuation))
alpha = re.compile('^[a-zA-Z_]+$')
alpha_or_num = re.compile('^[a-zA-Z_]+|[0-9_]+$')
alphanum = re.compile('^[a-zA-Z0-9_]+$')

frame_failed_count = 0
frame_success_count = 0


def main(args):
    usage = "%prog train.jsonlist output_dir"
    parser = OptionParser(usage=usage)
    parser.add_option('--label', dest='label', default=None,
                      help='field(s) to use as label (comma-separated): default=%default')
    parser.add_option('--test', dest='test', default=None,
                      help='Test data (test.jsonlist): default=%default')
    parser.add_option('--train-prefix', dest='train_prefix', default='train',
                      help='Output prefix for training data: default=%default')
    parser.add_option('--test-prefix', dest='test_prefix', default='test',
                      help='Output prefix for test data: default=%default')
    parser.add_option('--stopwords', dest='stopwords', default='snowball',
                      help='List of stopwords to exclude [None|mallet|snowball]: default=%default')
    parser.add_option('--min-doc-count', dest='min_doc_count', default=0,
                      help='Exclude words that occur in less than this number of documents')
    parser.add_option('--max-doc-freq', dest='max_doc_freq', default=1.0,
                      help='Exclude words that occur in more than this proportion of documents')
    parser.add_option('--keep-num', action="store_true", dest="keep_num", default=False,
                      help='Keep tokens made of only numbers: default=%default')
    parser.add_option('--keep-alphanum', action="store_true", dest="keep_alphanum", default=False,
                      help="Keep tokens made of a mixture of letters and numbers: default=%default")
    parser.add_option('--strip-html', action="store_true", dest="strip_html", default=False,
                      help='Strip HTML tags: default=%default')
    parser.add_option('--no-lower', action="store_true", dest="no_lower", default=False,
                      help='Do not lowercase text: default=%default')
    parser.add_option('--min-length', dest='min_length', default=3,
                      help='Minimum token length: default=%default')
    parser.add_option('--vocab-size', dest='vocab_size', default=None,
                      help='Size of the vocabulary (by most common, following above exclusions): default=%default')
    parser.add_option('--seed', dest='seed', default=42,
                      help='Random integer seed (only relevant for choosing test set): default=%default')

    (options, args) = parser.parse_args(args)

    train_infile = args[0]
    ref_dir = args[1]  # root directory for 'ref' references in input json
    output_dir = args[2]

    test_infile = options.test
    train_prefix = options.train_prefix
    test_prefix = options.test_prefix
    label_fields = options.label
    min_doc_count = int(options.min_doc_count)
    max_doc_freq = float(options.max_doc_freq)
    vocab_size = options.vocab_size
    stopwords = options.stopwords
    if stopwords == 'None':
        stopwords = None
    keep_num = options.keep_num
    keep_alphanum = options.keep_alphanum
    strip_html = options.strip_html
    lower = not options.no_lower
    min_length = int(options.min_length)
    seed = options.seed
    if seed is not None:
        np.random.seed(int(seed))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    preprocess_data(train_infile, ref_dir, test_infile, output_dir, train_prefix, test_prefix, min_doc_count, max_doc_freq, vocab_size, stopwords, keep_num, keep_alphanum, strip_html, lower,
                    min_length, label_fields=label_fields)


class Ref:
    def __init__(self, lines, name):
        self.words = []
        self.curr_idx = 0
        self.name = name
        for l in lines:
            self.add_line(l)

    def add_line(self, line):
        line = line.strip()
        if len(line) == 0:
            return
        data = line.split('\t')
        tokens, _ = tokenize(data[1])
        self.words.append(Word(data[1], data[2], data[3], data[4]))
        import re
        self.words.append(Word(re.sub('[^a-z]', '', data[1].lower()), data[2], data[3], data[4]))
        for token in tokens:
            w = Word(token, data[2], data[3], data[4])
            self.words.append(w)

    def get_frames(self):
        return [word.frame_type for word in self.words]

    def find_frame(self, word):
        start = self.curr_idx
        while self.curr_idx < len(self.words):
            w = self.words[self.curr_idx]
            self.curr_idx += 1
            if w.match(word):
                global frame_success_count
                frame_success_count += 1
                return w.frame_type
        # print("Warn: Failed to find frame for word " + word)
        global frame_failed_count
        frame_failed_count += 1
        self.curr_idx = start
        return '_'
        # return None

    def __str__(self):
        return "\n".join([str(w) for w in self.words])


class Word:
    def __init__(self, word, frame_word, frame_type, frame_arg):
        self.word = word
        self.frame_word = frame_word
        self.frame_type = frame_type
        self.frame_arg = frame_arg

    def match(self, word):
        me = self.word.lower()
        w = word.lower()
        if w == me:
            return True
        if me in w:
            return True
        return False

    def __str__(self):
        return self.word + '\t' + self.frame_type


def get_reference(directory, ref):
    lines = None
    with open(directory + os.sep + ref, "r", encoding="latin-1") as f:
        lines = f.readlines()
    r = Ref(lines, directory + os.sep + ref)
    return r


def preprocess_data(train_infile, ref_dir, test_infile, output_dir, train_prefix, test_prefix, min_doc_count=0, max_doc_freq=1.0, vocab_size=None, stopwords=None, keep_num=False, keep_alphanum=False,
                    strip_html=False, lower=True, min_length=3, label_fields=None):
    if stopwords == 'mallet':
        print("Using Mallet stopwords")
        stopword_list = fh.read_text(os.path.join('stopwords', 'mallet_stopwords.txt'))
    elif stopwords == 'snowball':
        print("Using snowball stopwords")
        stopword_list = fh.read_text(os.path.join('stopwords', 'snowball_stopwords.txt'))
    elif stopwords is not None:
        print("Using custom stopwords")
        stopword_list = fh.read_text(os.path.join('stopwords', stopwords + '_stopwords.txt'))
    else:
        stopword_list = []
    stopword_set = {s.strip() for s in stopword_list}

    print("Reading data files")
    train_items = fh.read_jsonlist(train_infile)
    n_train = len(train_items)
    print("Found {:d} training documents".format(n_train))

    if test_infile is not None:
        test_items = fh.read_jsonlist(test_infile)
        n_test = len(test_items)
        print("Found {:d} test documents".format(n_test))
    else:
        test_items = []
        n_test = 0

    ## DEBUG ONLY
    # train_items = train_items[0:200]
    # test_items = test_items[0:200]
    n_train = len(train_items)
    n_test = len(test_items)

    all_items = train_items + test_items
    n_items = n_train + n_test

    label_lists = {}
    if label_fields is not None:
        if ',' in label_fields:
            label_fields = label_fields.split(',')
        else:
            label_fields = [label_fields]
        for label_name in label_fields:
            label_set = set()
            for i, item in enumerate(all_items):
                if label_name is not None:
                    label_set.add(item[label_name])
            label_list = list(label_set)
            label_list.sort()
            n_labels = len(label_list)
            print("Found label %s with %d classes" % (label_name, n_labels))
            label_lists[label_name] = label_list
    else:
        label_fields = []

    # make vocabulary
    train_parsed = []
    test_parsed = []

    print("Parsing %d documents" % n_items)
    word_counts = Counter()
    doc_counts = Counter()
    frame_counts = Counter()
    count = 0

    vocab = None
    for i, item in enumerate(tqdm(all_items, desc="parsing")):
        if i % 1000 == 0 and count > 0:
            print(i)

        text = item['text']
        # ref = item['ref'] # reference file with frames
        ref = item['orig']
        ref = ref[ref.rfind(os.sep, 0, ref.rfind(os.sep)) + 1:]
        if i < n_train:
            ref = 'imdb-final-train' + os.sep + ref
        else:
            ref = 'imdb-final-test' + os.sep + ref
        tokens, _ = tokenize(text, strip_html=strip_html, lower=lower, keep_numbers=keep_num, keep_alphanum=keep_alphanum, min_length=min_length, stopwords=stopword_set, vocab=vocab)
        reference = get_reference(ref_dir, ref)

        # store the parsed documents
        if i < n_train:
            train_parsed.append((tokens, reference))
        else:
            test_parsed.append((tokens, reference))

        # keep track fo the number of documents with each word
        word_counts.update(tokens)
        # filter vocab by # docs with the word
        doc_counts.update(set(tokens))
        frame_counts.update(reference.get_frames())

    print("Size of full vocabulary=%d" % len(word_counts))

    print("Selecting the vocabulary")
    most_common = doc_counts.most_common()
    words, doc_counts = zip(*most_common)
    doc_freqs = np.array(doc_counts) / float(n_items)
    vocab = [word for i, word in enumerate(words) if doc_counts[i] >= min_doc_count and doc_freqs[i] <= max_doc_freq]
    most_common = [word for i, word in enumerate(words) if doc_freqs[i] > max_doc_freq]
    if max_doc_freq < 1.0:
        print("Excluding words with frequency > {:0.2f}:".format(max_doc_freq), most_common)

    print("Vocab size after filtering = %d" % len(vocab))
    if vocab_size is not None:
        if len(vocab) > int(vocab_size):
            vocab = vocab[:int(vocab_size)]

    vocab_size = len(vocab)
    print("Final vocab size = %d" % vocab_size)

    print("Most common words remaining:", ' '.join(vocab[:10]))
    vocab.sort()

    print("Size of full frames=%d" % len(frame_counts))
    # print(frame_counts)

    most_common = frame_counts.most_common()
    frame_names, doc_counts = zip(*most_common)
    frames = [frame for i, frame in enumerate(frame_names)]

    print("Most common frames:", ' '.join(frames[:10]))
    frames.sort()

    fh.write_to_json(vocab, os.path.join(output_dir, train_prefix + '.vocab.json'))
    fh.write_to_json(frames, os.path.join(output_dir, train_prefix + '.frames.json'))

    train_X_sage = process_subset(train_items, train_parsed, label_fields, label_lists, vocab, frames, output_dir, train_prefix)
    if n_test > 0:
        test_X_sage = process_subset(test_items, test_parsed, label_fields, label_lists, vocab, frames, output_dir, test_prefix)

    train_sum = np.array(train_X_sage.sum(axis=0))
    print("%d word-frame pairs missing from training data" % np.sum(train_sum == 0))

    if n_test > 0:
        test_sum = np.array(test_X_sage.sum(axis=0))
        print("%d word-frame pairs missing from test data" % np.sum(test_sum == 0))

    print("Done!")


def process_subset(items, parsed, label_fields, label_lists, vocab, frames, output_dir, output_prefix):
    n_items = len(items)
    vocab_size = len(vocab)
    frame_size = len(frames)
    vocab_index = dict(zip(vocab, range(vocab_size)))
    frame_index = dict(zip(frames, range(frame_size)))

    ids = []
    for i, item in enumerate(items):
        if 'id' in item:
            ids.append(item['id'])
    if len(ids) != n_items:
        ids = [str(i) for i in range(n_items)]

    # create a label index using string representations
    for label_field in label_fields:
        label_list = label_lists[label_field]
        n_labels = len(label_list)
        label_list_strings = [str(label) for label in label_list]
        label_index = dict(zip(label_list_strings, range(n_labels)))

        # convert labels to a data frame
        if n_labels > 0:
            label_matrix = np.zeros([n_items, n_labels], dtype=int)

            for i, item in enumerate(items):
                label = item[label_field]
                label_matrix[i, label_index[str(label)]] = 1

            labels_df = pd.DataFrame(label_matrix, index=ids, columns=label_list_strings)
            labels_df.to_csv(os.path.join(output_dir, output_prefix + '.' + label_field + '.csv'))

    X = np.zeros([n_items, vocab_size], dtype=int)
    F = np.zeros([n_items, frame_size], dtype=int)

    word_counter = Counter()
    frame_counter = Counter()
    mapper = {}
    # word_counter = Counter()
    doc_lines = []
    print("Converting to count representations")
    for i, (words, ref) in enumerate(tqdm(parsed, desc="processing subset")):
        # get the vocab indices of words that are in the vocabulary

        # print("w", words, len(words))
        word_indices = []
        frame_indices = []
        for word in words:
            frame = ref.find_frame(word)
            if frame is None:
                print(words)
                print(ref)
                print(ref.name)
                import sys
                sys.exit(1)
            if word in vocab_index and frame in frame_index:
                wi = vocab_index[word]
                fi = frame_index[frame]
                word_indices.append(wi)
                frame_indices.append(fi)
                if wi not in mapper:
                    mapper[wi] = Counter()
                mapper[wi].update([fi])
        # indices = [vocab_index[word] for word in words if word in vocab_index]
        # print("i", indices, len(indices))
        word_subset = [word for word in words if word in vocab_index]

        word_counter.clear()
        word_counter.update(word_indices)
        frame_counter.clear()
        frame_counter.update(frame_indices)
        # word_counter.clear()
        # word_counter.update(word_subset)

        if len(word_counter.keys()) > 0:
            # print(counter)
            # print(counter.keys())
            # print(counter.values())
            # print(np.ones(len(counter.keys()), dtype=int))
            # print(vocab_index)
            # print("VALUES\n", values, len(values))
            # 0,0,0... 1,1,1..., 2,2,2..., etc
            row_indexer = np.ones(len(word_counter.keys()), dtype=int) * i
            X[row_indexer, list(word_counter.keys())] += list(word_counter.values())

            row_indexer = np.ones(len(frame_counter.keys()), dtype=int) * i
            F[row_indexer, list(frame_counter.keys())] += list(frame_counter.values())

    # print("mapper: " + str(mapper))
    # for x in mapper.values():
    #     for y in x:
    #         if x[y] > 1:
    #             print(x, frames[y])
    ct = Counter()
    ct.update([len(x) for x in mapper.values()])

    fh.write_to_json(mapper, os.path.join(output_dir, output_prefix + '.framemap.json'))


    ratio_mapper = {}
    for x in mapper:
        sum = 0.0
        new_map = {}
        for y in mapper[x]:
            sum += mapper[x][y]
        for y in mapper[x]:
            new_map[y] = mapper[x][y] / sum
        ratio_mapper[x] = new_map
    fh.write_to_json(mapper, os.path.join(output_dir, output_prefix + '.framemap.json'))
    fh.write_to_json(ratio_mapper, os.path.join(output_dir, output_prefix + '.frameratio.json'))

    print("word - frame occurrences: " + str(ct))

    global frame_failed_count, frame_success_count
    print("Found frames in {:s} for {} of {} words ({}% success)"
          .format(output_prefix,
                  frame_success_count,
                  frame_success_count + frame_failed_count,
                  100.0 * frame_success_count / (frame_success_count + frame_failed_count)))
    frame_failed_count = 0
    frame_success_count = 0

    # convert to a sparse representation
    sparse_X = sparse.csr_matrix(X)
    fh.save_sparse(sparse_X, os.path.join(output_dir, output_prefix + '.npz'))

    sparse_F = sparse.csr_matrix(F)
    fh.save_sparse(sparse_F, os.path.join(output_dir, output_prefix + '.frames.npz'))

    print("Size of {:s} document-term matrix:".format(output_prefix), sparse_X.shape)
    print("Size of {:s} document-frame matrix:".format(output_prefix), sparse_F.shape)

    fh.write_to_json(ids, os.path.join(output_dir, output_prefix + '.ids.json'))

    sparse_X_sage = sparse.csr_matrix(X, dtype=float)

    return sparse_X_sage


def tokenize(text, strip_html=False, lower=True, keep_emails=False, keep_at_mentions=False, keep_numbers=False, keep_alphanum=False, min_length=3, stopwords=None, vocab=None):
    text = clean_text(text, strip_html, lower, keep_emails, keep_at_mentions)
    tokens = text.split()

    if stopwords is not None:
        tokens = ['_' if t in stopwords else t for t in tokens]

    # remove tokens that contain numbers
    if not keep_alphanum and not keep_numbers:
        tokens = [t if alpha.match(t) else '_' for t in tokens]

    # or just remove tokens that contain a combination of letters and numbers
    elif not keep_alphanum:
        tokens = [t if alpha_or_num.match(t) else '_' for t in tokens]

    # drop short tokens
    if min_length > 0:
        tokens = [t if len(t) >= min_length else '_' for t in tokens]

    counts = Counter()

    unigrams = [t for t in tokens if t != '_']
    counts.update(unigrams)

    if vocab is not None:
        tokens = [token for token in unigrams if token in vocab]
    else:
        tokens = unigrams

    return tokens, counts


def clean_text(text, strip_html=False, lower=True, keep_emails=False, keep_at_mentions=False):
    # remove html tags
    if strip_html:
        text = re.sub(r'<[^>]+>', '', text)
    else:
        # replace angle brackets
        text = re.sub(r'<', '(', text)
        text = re.sub(r'>', ')', text)
    # lower case
    if lower:
        text = text.lower()
    # eliminate email addresses
    if not keep_emails:
        text = re.sub(r'\S+@\S+', ' ', text)
    # eliminate @mentions
    if not keep_at_mentions:
        text = re.sub(r'\s@\S+', ' ', text)
    # replace underscores with spaces
    text = re.sub(r'_', ' ', text)
    # break off single quotes at the ends of words
    text = re.sub(r'\s\'', ' ', text)
    text = re.sub(r'\'\s', ' ', text)
    # remove periods
    text = re.sub(r'\.', '', text)
    # replace all other punctuation (except single quotes) with spaces
    text = replace.sub(' ', text)
    # remove single quotes
    text = re.sub(r'\'', '', text)
    # replace all whitespace with a single space
    text = re.sub(r'\s', ' ', text)
    # strip off spaces on either end
    text = text.strip()
    return text


if __name__ == '__main__':
    main(sys.argv[1:])
