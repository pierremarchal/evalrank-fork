#!/bin/bash

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

echo " "
currentDate=`date`
echo $currentDate
echo " "

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

#WORD_EMB_PATH='src/models/word_emb/toy_emb.txt' # 'src/models/word_emb/glove.840B.300d.txt'
WORD_EMB_PATH='src/models/word_emb/chive-1.1-mc90-aunit.txt'
#WORD_EMB_PATH='/Users/pierre/projects/sandbox_bias_analysis/models/cc.ja.300/cc.ja.300.vec'
EVAL_TYPE='similarity,ranking' # 'similarity', 'ranking'
DIST_METRIC='cos' # 'cos', 'l2'
BG_VOCAB='basic,wiki' # 'basic', 'wiki'
POST_PROCESS='False' # 'True', 'False'
OUTPUT='res.json'

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

python src/run_word_evaluation.py       \
        --word_emb_model                $WORD_EMB_PATH \
        --dist_metric                   $DIST_METRIC \
        --eval_type                     $EVAL_TYPE \
        --background_vocab_type         $BG_VOCAB \
        --post_process                  $POST_PROCESS \
        --output                        $OUTPUT \
        --skip_oov

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
