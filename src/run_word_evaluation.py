###
# Created Date: 2022-03-18 11:18:44
# Author: Bin Wang
# -----
# Copyright (c) 2022 National University of Singapore
# 
# -----
# Modified by pierremarchal
###

import argparse
import json
import logging
import sys

import w_data_loader
import w_evaluation
import w_model

logging.basicConfig(level=logging.DEBUG)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--word_emb_model", type=str, default=None,
                        help="name of word embedding model with full path (.txt file supported)")
    parser.add_argument("--dist_metric", type=str, default='cos',
                        help="distance measure between embeddings: cos, l2")
    parser.add_argument("--eval_type", type=str, default=None,
                        help="evaluation types: similarity,ranking")
    parser.add_argument("--background_vocab_type", type=str, default=None,
                        help="vocabulary used for background: basic, bccwj")
    parser.add_argument("--post_process", action="store_true",
                        help="whether to do post-processing on word embedding")
    parser.add_argument("--skip_oov", action='store_true',
                        help="skip test if source vector or target vector is missing; also ignore background words not"
                             "in the model")
    parser.add_argument("--output", type=str, default="res.json",
                        help="output file (JSON)")
    config = parser.parse_args()

    if config.dist_metric == 'cos':
        config.normalization = True
    elif config.dist_metric == 'l2':
        config.normalization = False
    else:
        sys.exit("Distance Metric NOT SUPPORTED: {}".format(config.dist_metric))

    config.eval_type = config.eval_type.split(',')
    config.background_vocab_type = config.background_vocab_type.split(',')
    config.centralization = True
    # display parameters
    logging.info("*** Parameters ***")
    for item, value in vars(config).items():
        logging.info("{}: {}".format(item, value))
    # load embedding model
    word_emb_model = w_model.WordEmbeddingModel(config)
    # load data
    word_pairs_data = w_data_loader.WordDatasetLoader(config, model=word_emb_model)
    # evaluation
    our_evaluator = w_evaluation.WordEmbEvaluator(config, word_pairs_data, word_emb_model)
    res_ws, res_rank = our_evaluator.eval()

    # write results to JSON file
    res = {
        "model": {
            "name": word_emb_model.name,
            "pathname": config.word_emb_model,
            "size": word_emb_model.word_emb.shape
        },
        "options": {
            "post-processing": config.post_process,
            "skip_oov": config.skip_oov,
            "distance_metric": config.dist_metric
        },
        "word_similarity": res_ws,
        "rankeval": res_rank,
    }
    with open(config.output, "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
