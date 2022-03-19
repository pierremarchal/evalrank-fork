#!/usr/bin/env python
# -*- coding:utf-8 -*-
###
# Created Date: 2022-03-18 11:18:44
# Author: Bin Wang
# -----
# Copyright (c) 2022 National University of Singapore
# 
# -----
# HISTORY:
# Date&Time 			By	Comments
# ----------			---	----------------------------------------------------------
###


import pdb
# here put the import lib
import os
import sys
import logging
import argparse

import data_loader
import models
import evaluation

logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.INFO)


if __name__ == "__main__":
    '''
        this is the main file for running word embedding evaluations for both similarity and ranking
    '''
    # - - - - - - - - - - - - - - - - -
    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--word_emb_model", type=str, default=None,
                        help="name of word embedding model with full path (.txt file supported)")
    parser.add_argument("--dist_metric", type=str, default='cos',
                        help="distance measure between embeddings: cos, l2")
    parser.add_argument("--eval_type", type=str, default=None,
                        help="evaluation types: similarity,ranking")
    config = parser.parse_args()

    # - - - - - - - - - - - - - - - - -
    if config.dist_metric == 'cos':
        config.normalization = True
    elif config.dist_metric == 'l2':
        config.normalization = False
    else:
        sys.exit("Distance Metric NOT SUPPORTED: {}".format(config.dist_metric))
    # - - - - - - - - - - - - - - - - -

    config.eval_type = config.eval_type.split(',')
    config.centralization = None
    
    
    if config.pos_pairs_type is not None: config.pos_pairs_type = config.pos_pairs_type.split(',')
    if config.background_vocab_type is not None: config.background_vocab_type = config.background_vocab_type.split(',')

    # display parameters
    logging.info("*** Parameters ***")
    for item, value in vars(config).items():
        logging.info("{}: {}".format(item, value))
    logging.info("")

    # - - - - - - - - - - - - - - - - -
    # run evaluation
    word_pairs_data = data_loader.Word_similarity_dataset_loader(config)
    word_emb_model  = models.Word_embedding_model(config)
    our_evaluator   = evaluation.Word_emb_evaluator(word_pairs_data, word_emb_model, config)
    ws_ori, ws_post, rank_results = our_evaluator.eval()



    '''
    # = = = = = save results = = = = = 
    path = './record/word/'

    if rank_results is not None:

        path_new = path + 'rank_wiki_synonym/'
        if not os.path.exists(path_new):
            os.makedirs(path_new)

        with open(path_new+'MR.txt', 'a') as f:
            f.write(str(config.model_index)+'\t'+str(rank_results['MR_ori'])+'\n')
        with open(path_new+'MR.txt', 'a') as f:
            f.write(str(config.model_index+1)+'\t'+str(rank_results['MR_post'])+'\n')

        with open(path_new+'MRR.txt', 'a') as f:
            f.write(str(config.model_index)+'\t'+str(rank_results['MRR_ori'])+'\n')
        with open(path_new+'MRR.txt', 'a') as f:
            f.write(str(config.model_index+1)+'\t'+str(rank_results['MRR_post'])+'\n')

        with open(path_new+'hits_1.txt', 'a') as f:
            f.write(str(config.model_index)+'\t'+str(rank_results['hits_1_ori'])+'\n')
        with open(path_new+'hits_1.txt', 'a') as f:
            f.write(str(config.model_index+1)+'\t'+str(rank_results['hits_1_post'])+'\n')

        with open(path_new+'hits_2.txt', 'a') as f:
            f.write(str(config.model_index)+'\t'+str(rank_results['hits_2_ori'])+'\n')
        with open(path_new+'hits_2.txt', 'a') as f:
            f.write(str(config.model_index+1)+'\t'+str(rank_results['hits_2_post'])+'\n')

        with open(path_new+'hits_3.txt', 'a') as f:
            f.write(str(config.model_index)+'\t'+str(rank_results['hits_3_ori'])+'\n')
        with open(path_new+'hits_3.txt', 'a') as f:
            f.write(str(config.model_index+1)+'\t'+str(rank_results['hits_3_post'])+'\n')
        
        with open(path_new+'hits_4.txt', 'a') as f:
            f.write(str(config.model_index)+'\t'+str(rank_results['hits_4_ori'])+'\n')
        with open(path_new+'hits_4.txt', 'a') as f:
            f.write(str(config.model_index+1)+'\t'+str(rank_results['hits_4_post'])+'\n')
        
        with open(path_new+'hits_5.txt', 'a') as f:
            f.write(str(config.model_index)+'\t'+str(rank_results['hits_5_ori'])+'\n')
        with open(path_new+'hits_5.txt', 'a') as f:
            f.write(str(config.model_index+1)+'\t'+str(rank_results['hits_5_post'])+'\n')
        
        with open(path_new+'hits_6.txt', 'a') as f:
            f.write(str(config.model_index)+'\t'+str(rank_results['hits_6_ori'])+'\n')
        with open(path_new+'hits_6.txt', 'a') as f:
            f.write(str(config.model_index+1)+'\t'+str(rank_results['hits_6_post'])+'\n')
        
        with open(path_new+'hits_7.txt', 'a') as f:
            f.write(str(config.model_index)+'\t'+str(rank_results['hits_7_ori'])+'\n')
        with open(path_new+'hits_7.txt', 'a') as f:
            f.write(str(config.model_index+1)+'\t'+str(rank_results['hits_7_post'])+'\n')
       
        with open(path_new+'hits_8.txt', 'a') as f:
            f.write(str(config.model_index)+'\t'+str(rank_results['hits_8_ori'])+'\n')
        with open(path_new+'hits_8.txt', 'a') as f:
            f.write(str(config.model_index+1)+'\t'+str(rank_results['hits_8_post'])+'\n')
        
        with open(path_new+'hits_9.txt', 'a') as f:
            f.write(str(config.model_index)+'\t'+str(rank_results['hits_9_ori'])+'\n')
        with open(path_new+'hits_9.txt', 'a') as f:
            f.write(str(config.model_index+1)+'\t'+str(rank_results['hits_9_post'])+'\n')
        
        with open(path_new+'hits_10.txt', 'a') as f:
            f.write(str(config.model_index)+'\t'+str(rank_results['hits_10_ori'])+'\n')
        with open(path_new+'hits_10.txt', 'a') as f:
            f.write(str(config.model_index+1)+'\t'+str(rank_results['hits_10_post'])+'\n')
        
        with open(path_new+'hits_11.txt', 'a') as f:
            f.write(str(config.model_index)+'\t'+str(rank_results['hits_11_ori'])+'\n')
        with open(path_new+'hits_11.txt', 'a') as f:
            f.write(str(config.model_index+1)+'\t'+str(rank_results['hits_11_post'])+'\n')
        
        with open(path_new+'hits_12.txt', 'a') as f:
            f.write(str(config.model_index)+'\t'+str(rank_results['hits_12_ori'])+'\n')
        with open(path_new+'hits_12.txt', 'a') as f:
            f.write(str(config.model_index+1)+'\t'+str(rank_results['hits_12_post'])+'\n')
        
        with open(path_new+'hits_13.txt', 'a') as f:
            f.write(str(config.model_index)+'\t'+str(rank_results['hits_13_ori'])+'\n')
        with open(path_new+'hits_13.txt', 'a') as f:
            f.write(str(config.model_index+1)+'\t'+str(rank_results['hits_13_post'])+'\n')
        
        with open(path_new+'hits_14.txt', 'a') as f:
            f.write(str(config.model_index)+'\t'+str(rank_results['hits_14_ori'])+'\n')
        with open(path_new+'hits_14.txt', 'a') as f:
            f.write(str(config.model_index+1)+'\t'+str(rank_results['hits_14_post'])+'\n')
        
        with open(path_new+'hits_15.txt', 'a') as f:
            f.write(str(config.model_index)+'\t'+str(rank_results['hits_15_ori'])+'\n')
        with open(path_new+'hits_15.txt', 'a') as f:
            f.write(str(config.model_index+1)+'\t'+str(rank_results['hits_15_post'])+'\n')


    '''


