# Prerequisites
Note: Other versions of python and packages below are not tested. But we believe they should work as long as python3+ is used.

Environment
* python (3.6.6)

Mandatory Packages
* numpy (1.15.4)

* scipy (1.1.0)

* matplotlib (3.0.2)

* gensim (3.6.0)

All of the above can be installed via `pip`, e.g.,
```
pip install 'numpy==1.15.4'
```

Optional Packages (if use GPU)
* cupy

Assume available cuda version is 9.0. Install it by
```
pip install cupy-cuda90
```
More details can be found at [Chainer](https://cupy.chainer.org/). Also, do not forget to append the CUDA paths in bash environment. The following is a working example:
```
CUDA_PATH=/path/to/cuda-9.0.176
CUDNN_PATH=/path/to/cudnn-9.0-linux-x64-v7.0-rc
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$CUDA_PATH/extras/CUPTI/lib64:$CUDNN_PATH/lib64:$LD_LIBRARY_PATH
export PATH=$CUDA_PATH/bin:$PATH
```
Feel free to test other cuda versions.

# Experiments
The following will produce results for BLI among 6 European languages.
(1) Download the [fasttext](https://fasttext.cc/docs/en/pretrained-vectors.html) embeddings and dictionaries.
```
./get_data.sh
```
A ./data directory will be created. Under that are embeddings for 6 European languages, de (German), en (English), es (Spanish), fr (French), it (Italian) and pt (Portuguese), as well as dictionaries for all the pairs.

(2) Get translation accuracy for a `src`-`tgt` pair, using a specified retrieval `method` (one of {nn, isf, csls, hnn}). Run
```
./bli_exp.sh $src $tgt $method
```
The experiment follows the "supervised" setup at [MUSE](https://github.com/facebookresearch/MUSE), but differs in that we use larger test dictionaries (data/`src`-`tgt`.txt). The output is a log, `/exp/BLI/src-tgt.method.log`. By default, we use 500K as the vocabulary size for both source and target languages.

To see all translated words, set `save_translation=1` in `bli_exp.sh`. The translated words will be stored in `./exp/BLI/src-tgt.method.trans`. Each row of the file is a source word, followed by top-10 candidates of translation, from the "best" to the "worst".

(3) Check how hubness is reduced (Figure 4 and Table 2). For example, to check the hubness for Portuguese-to-English task, simply run
```
python hubness_in_translations.py pt en -k 5 -N 200
```
It will produce `k-occurrence` (`k=5` in this case) histograms, as measures of hubness, for the different methods. In particular, long tail of the histogram indicates strong hubness, which should be reduced. The Portuguese to English example will have the following histograms, where HNN has the shortest tail, *i.e.*, weakest hubness.
<p align="center">
    <img src="exp/bli_500K/pt-en.k_occur.png" width="400">
</p>
We will also see some (200 in this case) "hubby" words being listed, for example:

| "hubby" words |   NN  | ISF | CSLS | HNN |
|:-------------:|:-----:|:---:|:----:|:---:|
|   conspersus  | 1,776 |   0 |  374 |   0 |
|      s+bd     |   912 |   7 |  278 |  16 |
|      were     |   798 |  99 |  262 |  24 |
|      you      |   474 |  12 |   57 |  20 |

The numerics are the number of times these words being retrieved. A big value indicates that the word is a hub. Note how the values are reduced by HNN.

[[1]](http://www.jmlr.org/papers/volume11/radovanovic10a/radovanovic10a.pdf) Milos Radovanovic, Alexandros Nanopoulos, and Mirjana Ivanovic. 2010. Hubs in space: Popular nearest neighbors in high-dimensional data. Journal of Machine Learning Research.

[[2]](https://arxiv.org/pdf/1702.03859.pdf) Samuel L. Smith, David H. P. Turban, Steven Hamblin, and Nils Y. Hammerla. 2017. Offline bilingual word vectors, orthogonal transformations and the inverted softmax. In International Conference on Learning Representations.

[[3]](https://arxiv.org/pdf/1710.04087.pdf) Alexis Conneau, Guillaume Lample, Marc' Aurelio Ranzato, Ludovic Denoyer, and Hervé Jégou. 2018. Word translation without parallel data. In International Conference on Learning Representations.
