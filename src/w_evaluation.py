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
from collections import defaultdict

import numpy as np
from prettytable import PrettyTable
from scipy.stats.stats import kendalltau
from scipy.stats.stats import pearsonr
from scipy.stats.stats import spearmanr


class WordEmbEvaluator:
    """
    run evaluation by similarity and ranking
    """

    def __init__(self, config, word_pairs_data, word_emb_model) -> None:
        """ define what tasks to perform in this module """
        self.eval_by_ranking = 'ranking' in config.eval_type
        self.eval_by_similarity = 'similarity' in config.eval_type
        self.dist_metric = config.dist_metric
        self.skip_oov = config.skip_oov
        self.word_pairs_data = word_pairs_data
        self.word_emb_model = word_emb_model

    def eval(self):
        """ main functions for this module """
        res_rank = None
        if self.eval_by_ranking:
            logging.info('*** Evaluation on word ranking tasks ***')
            table_rank, res_rank = self.eval_for_ranking()
            logging.info("\n" + str(table_rank))

        res_ws = None
        if self.eval_by_similarity:
            logging.info('*** Evaluation on word similarity tasks ***')
            table_ws, res_ws = self.eval_for_similarity()
            logging.info("\n" + str(table_ws))

        return res_ws, res_rank

    def eval_for_ranking(self):
        """ evaluate the embeddings on ranking task """

        res_rank = dict()
        ranks = []
        ranks_by_dataset = defaultdict(list)
        test_count = 0
        test_count_by_dataset = defaultdict(int)
        oov_count = 0
        oov_count_by_dataset = defaultdict(int)
        vocab_embs = [self.word_emb_model.compute_embedding(word)[0] for word in self.word_pairs_data.vocab]
        vocab_embs = np.stack(vocab_embs)

        res_by_dataset = defaultdict(list)
        for w1, w2 in self.word_pairs_data.pos_pairs:
            dataset_name = self.word_pairs_data.pair2dataset[(w1, w2)][0]
            test_info = dict()
            w1_emb, is_source_oov = self.word_emb_model.compute_embedding(w1)
            w2_emb, is_target_oov = self.word_emb_model.compute_embedding(w2)
            test_info["source"] = w1
            test_info["is_source_oov"] = is_source_oov
            test_info["target"] = w2
            test_info["is_target_oov"] = is_target_oov
            test_info["skipped"] = False
            logging.debug(f"evalrank ({dataset_name}): {w1} (is_oov={is_source_oov}); {w2} (is_oov={is_target_oov})")

            test_count += 1
            test_count_by_dataset[dataset_name] += 1

            if is_source_oov or is_target_oov:
                oov_count += 1
                oov_count_by_dataset[dataset_name] += 1
                if self.skip_oov:
                    logging.info(f"skip test (missing vector); oov_count={oov_count}")
                    test_info["skipped"] = True
                    res_by_dataset[dataset_name].append(test_info)
                    continue

            if self.dist_metric == 'cos':
                scores = np.dot(vocab_embs, w1_emb)
            else:
                assert self.dist_metric == 'l2'
                scores = 1 / (np.linalg.norm((vocab_embs - w1_emb), axis=1) + 1)

            background_indexes = np.argsort(scores)[::-1]
            ranked = [self.word_pairs_data.vocab[i] for i in background_indexes]
            rank = ranked.index(w2)

            if rank == 0:
                rank = 1
            ranks.append(int(rank))
            ranks_by_dataset[dataset_name].append(int(rank))

            top10 = [self.word_pairs_data.vocab[i] for i in background_indexes[1:11]]
            test_info["most_similar"] = top10
            test_info["rank"] = int(rank)
            logging.debug(f"rank={rank}; top10={top10}")
            res_by_dataset[dataset_name].append(test_info)

        logging.info(f"oov ratio = {oov_count / test_count * 100:.2f}% ({oov_count} / {test_count})")

        mr = np.mean(ranks)
        mrr = np.mean(1. / np.array(ranks))

        res_rank["test_count"] = test_count
        res_rank["skipped_test_count"] = oov_count
        res_rank["MR"] = mr
        res_rank["MRR"] = mrr
        res_rank["hits_at_k"] = list()

        hits_max_bound = 10
        for i in range(hits_max_bound):
            score = sum(np.array(ranks) <= (i + 1)) / len(ranks)
            res_rank["hits_at_k"].append(score)

        data = defaultdict(dict)
        for name, ds_ranks in ranks_by_dataset.items():
            data[name]["test_count"] = test_count_by_dataset[name]
            data[name]["skipped_test_count"] = oov_count_by_dataset[name]
            data[name]["MR"] = np.mean(ds_ranks)
            data[name]["MRR"] = np.mean(1. / np.array(ds_ranks))
            data[name]["hits_at_k"] = list()
            for i in range(hits_max_bound):
                score = sum(np.array(ds_ranks) <= (i + 1)) / len(ds_ranks)
                data[name]["hits_at_k"].append(score)
            data[name]["details"] = res_by_dataset[name]

        res_rank["datasets"] = data

        table = PrettyTable(['Scores', 'Emb'])
        table.add_row(['MR', mr])
        table.add_row(['MRR', mrr])
        for i in [0, 2]:
            table.add_row(['Hits@' + str(i + 1), res_rank['hits_at_k'][i]])

        return table, res_rank

    def eval_for_similarity(self):
        """ evaluate the embeddings on similarity task """

        res_ws = {}

        oov_count_by_dataset = defaultdict(int)
        test_count_by_dataset = defaultdict(int)

        all_predicts = []
        all_expected = []

        results_by_dataset = dict()
        details_by_dataset = defaultdict(list)
        for dataset_name, data_pairs in self.word_pairs_data.ws_data.items():
            predicts = []
            expected = []
            for w1, w2, sc in data_pairs:
                test_info = dict()
                w1_emb, is_source_oov = self.word_emb_model.compute_embedding(w1)
                w2_emb, is_target_oov = self.word_emb_model.compute_embedding(w2)
                test_info["source"] = w1
                test_info["is_source_oov"] = is_source_oov
                test_info["target"] = w2
                test_info["is_target_oov"] = is_target_oov
                test_info["skipped"] = False
                logging.debug(f"similarity: {w1} (oov? => {is_source_oov}) <=> {w2} (oov? => {is_target_oov})")

                test_count_by_dataset[dataset_name] += 1
                if is_source_oov or is_target_oov:
                    oov_count_by_dataset[dataset_name] += 1
                    if self.skip_oov:
                        logging.info(f"skip test (missing vector)")
                        test_info["skipped"] = True
                        details_by_dataset[dataset_name].append(test_info)
                        continue

                if self.dist_metric == 'cos':
                    predict = w1_emb.dot(w2_emb.transpose())
                else:
                    assert self.dist_metric == 'l2'
                    predict = 1 / (np.linalg.norm(w1_emb - w2_emb) + 1)  # note 1/(1+d)

                test_info["expected"] = float(sc)
                test_info["actual"] = float(predict)
                logging.debug(f"predict={predict}; expect={sc}")
                predicts.append(predict)
                expected.append(sc)
                details_by_dataset[dataset_name].append(test_info)

            results_by_dataset[dataset_name] = {
                "test_count": test_count_by_dataset[dataset_name],
                "skipped_test_count": oov_count_by_dataset[dataset_name],
                'pearson': pearsonr(predicts, expected)[0],
                'spearman': spearmanr(predicts, expected)[0],
                'kendall': kendalltau(predicts, expected)[0],
                'details': details_by_dataset[dataset_name]
            }

            all_predicts += predicts
            all_expected += expected

        res_ws["test_count"] = sum(test_count_by_dataset.values())
        res_ws["skipped_test_count"] = sum(oov_count_by_dataset.values())
        res_ws["pearson"] = pearsonr(all_predicts, all_expected)[0]
        res_ws["spearman"] = spearmanr(all_predicts, all_expected)[0]
        res_ws["kendall"] = kendalltau(all_predicts, all_expected)[0]
        res_ws["datasets"] = results_by_dataset

        table = PrettyTable(['Index', 'METHOD', 'DATASET', 'Pearson', 'Spearman', 'Kendall'])
        count = 1
        for dataset, resu in res_ws["datasets"].items():
            table.add_row([count, 'Emb', dataset,
                           res_ws["datasets"][dataset]['pearson'],
                           res_ws["datasets"][dataset]['spearman'],
                           res_ws["datasets"][dataset]['kendall']])
            count += 1

        return table, res_ws
