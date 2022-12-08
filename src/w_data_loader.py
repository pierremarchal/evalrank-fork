###
# Created Date: 2022-03-19 14:00:56
# Author: Bin Wang
# -----
# Copyright (c) 2022 National University of Singapore
# 
# -----
# Modified by pierremarchal
###


import logging
import os.path
from collections import defaultdict

POS_PAIRS_FILENAME = 'pos_pair_jsim.txt'
WS_DATA_PATH = "./data/word_similarity/"
WS_DATASET_NAMES = [
    'jsim_adj_new_strict.txt',
    'jsim_adv_new_strict.txt',
    'jsim_noun_new_strict.txt',
    'jsim_verb_new_strict.txt',
]
RANK_DATA_PATH = "./data/word_evalrank/"
BASIC_VOCAB = "jsim_vocab.txt"
BCCWJ_VOCAB = "bccwj_vocab.txt"


class WordDatasetLoader:
    """
    dataset loader for word similarity tasks
    """

    def __init__(self, config, model=None) -> None:
        """ initialization """
        logging.info("*** Data Preparation ***")
        self.config = config
        self.skip_oov = config.skip_oov
        self.model = model
        self.pair2dataset = defaultdict(list)  # (w1, w2): [ds1, ds2, ...]
        # load data for word similarity
        if 'similarity' in config.eval_type:
            self.ws_data = {}  # word similarity data
            logging.info("Loading {} Word Similarity Datasets".format(len(WS_DATASET_NAMES)))
            self.load_word_similarity_dataset()
            logging.info("Finished")
        # load data for evalrank
        if 'ranking' in config.eval_type:
            self.pos_pairs = []  # ranking: pos pairs
            self.vocab = []  # ranking: background vocab
            logging.info("Loading Similar Word Pairs for Ranking")
            self.load_pos_pairs()
            self.pos_pairs.sort()
            logging.info("Loading Background Vocab for Ranking")
            self.build_basic_vocab()
            self.build_more_vocab()

    def load_word_similarity_dataset(self):
        """ load word similarity datasets (e.g. {'EN-WS-353-ALL.txt': [['book', 'paper', 5.25]]} """
        for dataset_name in WS_DATASET_NAMES:
            full_dataset_path = os.path.join(WS_DATA_PATH, dataset_name)
            cur_dataset = []
            with open(full_dataset_path) as f:
                for line in f:
                    x, y, sim_score = line.strip().split()
                    self.pair2dataset[(x, y)].append(dataset_name)
                    cur_dataset.append([x, y, float(sim_score)])
            self.ws_data[dataset_name] = cur_dataset

    def load_pos_pairs(self):
        """ collect positive pairs from word similarity dataset """
        logging.info("Top 25% from Word Similarity Dataset are used as Positive Pairs")
        with open(os.path.join(RANK_DATA_PATH, POS_PAIRS_FILENAME), 'r') as f:
            lines = f.readlines()
            for line in lines:
                cur_line = line.strip().split('\t')
                self.pos_pairs.append(cur_line)
        logging.info("{} Positive Pairs from Word Similarity Datasets".format(len(self.pos_pairs)))

    def add_to_vocab(self, word) -> None:
        if word in self.vocab:
            pass
        elif not self.skip_oov:
            self.vocab.append(word)
        elif not self.model:
            self.vocab.append(word)
        elif word in self.model.vocab:
            self.vocab.append(word)

    def build_basic_vocab(self):
        """ build basic vocabulary from positive pairs """
        for item in self.pos_pairs:
            self.add_to_vocab(item[0])
            self.add_to_vocab(item[1])

        logging.info("Background Vocab Collected from Similar Word Pairs; v-size={}".format(len(self.vocab)))

    def build_more_vocab(self):
        """ build more vocabulary from doc """
        if 'basic' in self.config.background_vocab_type:
            with open(os.path.join(RANK_DATA_PATH, BASIC_VOCAB), 'r') as f:
                lines = f.readlines()
                for line in lines:
                    cur_line = line.strip()
                    self.add_to_vocab(cur_line)
            logging.info("Background Vocab Collected from Word Similarity Datasets; v-size={}".format(len(self.vocab)))
        if 'bccwj' in self.config.background_vocab_type:
            with open(os.path.join(RANK_DATA_PATH, BCCWJ_VOCAB), 'r') as f:
                lines = f.readlines()
                for line in lines:
                    cur_line = line.strip()
                    self.add_to_vocab(cur_line)
            logging.info("Background Vocab Collected from BCCWJ; v-size={}".format(len(self.vocab)))
