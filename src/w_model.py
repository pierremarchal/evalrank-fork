###
# Created Date: 2022-03-19 14:46:46
# Author: Bin Wang
# -----
# Copyright (c) 2022 National University of Singapore
# 
# -----
# Modified by pierremarchal
###

import io
import logging
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA


class WordEmbeddingModel:
    """
    wrapper for word embedding models
    """

    def __init__(self, config) -> None:
        """ initializable for word embedding model, save vocab, embedding """
        self.word_emb_path = config.word_emb_model
        self.name = Path(config.word_emb_model).stem

        logging.info("*** Word Embedding Model Initialization ***")
        self.word2id = {}
        self.vocab = None
        self.word_emb = None
        self.word_emb_avg = None
        self.wvec_dim = None
        self.min_dims = 25

        self.read_word_embs()

        if config.post_process:
            self.post_processing_embs()

        self.wvec_dim = self.word_emb.shape[1]

        if config.centralization:
            self.word_emb = self.word_emb - self.word_emb.mean(axis=0, keepdims=True)
            self.word_emb_avg = np.mean(self.word_emb, axis=0)

        if config.normalization:
            self.normalizing_word_vectors()

    def read_word_embs(self):
        """ read the original word embedding """

        file_path = self.word_emb_path
        fin = io.open(file_path, 'r', encoding='utf-8', newline='\n', errors='ignore')

        count = 0
        exists = 0
        word2id = {}
        word_embs = []

        for line in fin:
            tokens = line.rstrip().split(' ')
            # skip if not long enough
            if len(tokens) < self.min_dims:
                logging.warning('Not enough embeddings for current word: {}'.format(tokens))
                continue

            if tokens[0] not in word2id:
                word2id[tokens[0]] = count
                word_embs.append(np.fromiter(map(float, tokens[1:]), dtype=np.float32))
                count += 1
            else:
                exists += 1

        word_embs_np = np.stack(word_embs)

        logging.info('{} words appeared more than twice'.format(exists))
        logging.info('Embedding set: {} words with {} dimensions'.format(word_embs_np.shape[0], word_embs_np.shape[1]))

        self.word2id = word2id
        self.vocab = [*self.word2id]
        self.word_emb = word_embs_np
        self.word_emb_avg = np.mean(word_embs_np, axis=0)

    def post_processing_embs(self):
        """ compute post-processing embedding (Principal component removal) """

        pp_comp = 2

        ori_word_emb = self.word_emb
        word_emb_np_tilda = ori_word_emb - np.mean(ori_word_emb, axis=0)
        u = PCA(n_components=pp_comp).fit(word_emb_np_tilda).components_
        new_word_emb_np = word_emb_np_tilda - (word_emb_np_tilda @ u.T @ u)

        logging.info('Embedding set: {} words with {} dimensions after '
                     'post processing'.format(new_word_emb_np.shape[0], new_word_emb_np.shape[1]))

        self.word_emb = new_word_emb_np
        self.word_emb_avg = np.mean(new_word_emb_np, axis=0)

    def normalizing_word_vectors(self):
        """ normalizing word vectors """
        logging.info('Normalizing word vectors')
        self.word_emb = self.word_emb / np.linalg.norm(self.word_emb, axis=1)[:, np.newaxis]
        self.word_emb_avg = np.mean(self.word_emb, axis=0)

    def compute_embedding(self, word):
        """ return embedding when word provided """
        # return average vector if not in the database
        if word in self.vocab:
            index = self.word2id[word]
            word_emb = self.word_emb[index]
            oov = False
        else:
            word_emb = self.word_emb_avg
            oov = True

        return word_emb, oov
