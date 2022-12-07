#!/usr/bin/env python
# -*- coding:utf-8 -*-
###
# Created Date: 2022-03-19 16:45:21
# Author: Bin Wang
# -----
# Copyright (c) 2022 National University of Singapore
# 
# -----
# Modified by pierremarchal
###

import logging
import sys

import numpy as np
from prettytable import PrettyTable
from scipy.stats.stats import kendalltau
from scipy.stats.stats import pearsonr
from scipy.stats.stats import spearmanr
from tqdm import tqdm


class Word_emb_evaluator:
    ''' run evaluation by similarity and ranking '''

    def __init__(self, config, word_pairs_data, word_emb_model) -> None:
        ''' define what tasks to perform in this module '''

        self.eval_by_ranking = 'ranking' in config.eval_type
        self.eval_by_similarity = 'similarity' in config.eval_type
        self.dist_metric = config.dist_metric
        self.skip_oov = config.skip_oov

        self.word_pairs_data = word_pairs_data
        self.word_emb_model = word_emb_model

    def eval(self):
        ''' main functions for this module '''

        if self.eval_by_ranking:
            logging.info('')
            logging.info('*** Evaluation on word ranking tasks ***')
            table_rank, res_rank = self.eval_for_ranking()
            logging.info("\n" + str(table_rank))

        if self.eval_by_similarity:
            logging.info('')
            logging.info('*** Evaluation on word similarity tasks ***')
            table_ws, res_ws = self.eval_for_similarity()
            logging.info("\n" + str(table_ws))

        return res_ws, res_rank

    def eval_for_ranking(self):
        """ evaluate the embeddings on ranking task """

        ranks = []
        test_count = 0
        oov_count = 0
        vocab_embs = [self.word_emb_model.compute_embedding(word)[0] for word in self.word_pairs_data.vocab]
        vocab_embs = np.stack(vocab_embs)

        # for pair in tqdm(self.word_pairs_data.pos_pairs, leave=False):
        for pair in self.word_pairs_data.pos_pairs:

            w1, w2 = pair
            w1_emb, is_source_oov = self.word_emb_model.compute_embedding(w1)
            w2_emb, is_target_oov = self.word_emb_model.compute_embedding(w2)

            test_count += 1
            if is_source_oov or is_target_oov:
                oov_count += 1
                if self.skip_oov:
                    logging.info(f"skip test (missing vector)")
                    continue

            if self.dist_metric == 'cos':

                pos_score = np.dot(w1_emb, w2_emb)
                bg_scores = np.dot(vocab_embs, w1_emb)
                background_scores = np.sort(bg_scores)[::-1]
                background_indexes = np.argsort(bg_scores)[::-1]

            elif self.dist_metric == 'l2':

                pos_score = 1 / (np.linalg.norm(w1_emb - w2_emb) + 1)
                bg_scores = 1 / (np.linalg.norm((vocab_embs - w1_emb), axis=1) + 1)
                background_scores = np.sort(bg_scores)[::-1]
                background_indexes = np.argsort(bg_scores)[::-1]

            else:
                sys.exit("Distance Metric NOT SUPPORTED: {}".format(self.dist_metric))

            logging.debug(f"rankeval: {w1} (oov? => {is_source_oov}) <=> {w2} (oov? => {is_target_oov})")
            logging.debug(f"similarity({w1},{w2}) = {pos_score}")

            rank = len(background_scores) - np.searchsorted(background_scores[::-1], pos_score, side='right')

            if rank == 0:
                rank = 1
            ranks.append(int(rank))
            logging.debug(f"rank={rank}")
            top3 = [self.word_pairs_data.vocab[i] for i in background_indexes[1:4]]
            logging.debug(f"top3={top3}")

        logging.debug(f"oov ratio = {oov_count / test_count * 100:.2f}% ({oov_count} / {test_count})")

        MR = np.mean(ranks)
        MRR = np.mean(1. / np.array(ranks))

        hits_scores = []
        hits_max_bound = 15

        for i in range(hits_max_bound):
            hits_scores.append(sum(np.array(ranks) <= (i + 1)) / len(ranks))

        res_rank = {'MR': MR,
                    'MRR': MRR}

        for i in range(hits_max_bound): res_rank['hits_' + str(i + 1)] = hits_scores[i]

        table = PrettyTable(['Scores', 'Emb'])
        table.add_row(['MR', MR])
        table.add_row(['MRR', MRR])

        for i in range(hits_max_bound):
            if i in [0, 2]:
                table.add_row(['Hits@' + str(i + 1), res_rank['hits_' + str(i + 1)]])

        return table, res_rank

    def eval_for_similarity(self):
        ''' evaluate the embeddings on similarity task '''

        results = {}

        total_oov_count = 0
        total_test_count = 0

        all_predicts = []
        all_expected = []

        #for dataset_name, data_pairs in tqdm(self.word_pairs_data.ws_data.items(), leave=False):
        for dataset_name, data_pairs in self.word_pairs_data.ws_data.items():
            predicts = []
            expected = []

            oov_count = 0
            test_count = 0

            for w1, w2, sc in data_pairs:
                w1_emb, is_source_oov = self.word_emb_model.compute_embedding(w1)
                w2_emb, is_target_oov = self.word_emb_model.compute_embedding(w2)

                logging.debug(f"similarity: {w1} (oov? => {is_source_oov}) <=> {w2} (oov? => {is_target_oov})")

                test_count += 1
                if is_source_oov or is_target_oov:
                    oov_count += 1
                    if self.skip_oov:
                        logging.info(f"skip test (missing vector)")
                        continue

                if self.dist_metric == 'cos':
                    predict = w1_emb.dot(w2_emb.transpose())
                elif self.dist_metric == 'l2':
                    predict = 1 / (np.linalg.norm(w1_emb - w2_emb) + 1)  # note 1/(1+d)
                else:
                    sys.exit("Distance Metric NOT SUPPORTED: {}".format(self.dist_metric))

                logging.debug(f"predict={predict}; expect={sc}")

                predicts.append(predict)
                expected.append(sc)

            pearsonr_res = pearsonr(predicts, expected)[0]
            spearmanr_res = spearmanr(predicts, expected)[0]
            kendall_res = kendalltau(predicts, expected)[0]

            results[dataset_name] = {'Pearson Corr': pearsonr_res,
                                     'Spearman Corr': spearmanr_res,
                                     'Kendall Corr': kendall_res}

            total_oov_count += oov_count
            total_test_count += test_count
            all_predicts += predicts
            all_expected += expected

        #results['all'] = {
        #    'Pearson Corr': pearsonr(all_predicts, all_expected)[0],
        #    'Spearman Corr': spearmanr(all_predicts, all_expected)[0],
        #    'Kendall Corr': kendalltau(all_predicts, all_expected)[0]
        #}

        table = PrettyTable(['Index', 'METHOD', 'DATASET', 'Pearson', 'Spearman', 'Kendall'])
        count = 1
        for dataset, resu in results.items():
            table.add_row([count, 'Emb', dataset, resu['Pearson Corr'], resu['Spearman Corr'], resu['Kendall Corr']])
            count += 1

        return table, results
