export PIP_INDEX_URL=https://pypi.python.org/simple
source /mnt/home/jiajihuang/envs/py3.6.6-OT_matching/bin/activate
export DICT_DIR=/mnt/data/jiajihuang/bilingual_lexicon_induction/dictionaries/
export EMB_DIR=/mnt/data/jiajihuang/bilingual_lexicon_induction/embeddings/

CUDA_PATH=/tools/cuda-9.0.176
CUDNN_PATH=/tools/cudnn-9.0-linux-x64-v7.0-rc
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$CUDA_PATH/extras/CUPTI/lib64:$CUDNN_PATH/lib64:$LD_LIBRARY_PATH
export PATH=$CUDA_PATH/bin:$PATH
