#!/bin/bash

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

echo " "
currentDate=`date`
echo $currentDate
echo " "

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

WORD_EMB_PATH='/Users/pierre/projects/sandbox_bias_analysis/models/chive-1.1-mc90-aunit/chive-1.1-mc90-aunit.txt'
EVAL_TYPE='similarity,ranking' # 'similarity', 'ranking'
DIST_METRIC='cos' # 'cos', 'l2'
BG_VOCAB='basic,bccwj' # 'basic', 'bccwj'
OUTPUT='res.json'

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

python src/run_word_evaluation.py       \
        --word_emb_model                $WORD_EMB_PATH \
        --dist_metric                   $DIST_METRIC \
        --eval_type                     $EVAL_TYPE \
        --background_vocab_type         $BG_VOCAB \
        --output                        $OUTPUT \
        --skip_oov \
        #--post_process
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
