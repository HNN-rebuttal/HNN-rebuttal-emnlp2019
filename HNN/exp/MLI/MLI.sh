#!/bin/bash
# Monolingual Lexicon Induction, plot accuracy over 
data_root=/mnt/data/jiajihuang/fasttext/
split_dir=varyS_V_500K_Bins_50

th=1.0
sbatch -p TitanXx8 --gres=gpu:1 --wrap "python ../../bilingual_lexicon_induction.py $data_root/wiki-news-300d-1M.vec $data_root/crawl-300d-2M.vec --src_vocab_file $data_root/$split_dir/vocab --tgt_vocab_file $data_root/$split_dir/vocab -d $data_root/$split_dir/S_10K/all.seed -t $data_root/$split_dir/??.test --loss_type hinge --orth --hinge_lr 1e-3 --hinge_epoch 5 --hinge_ns 10 --hinge_batch 64 --hinge_th $th --hinge_sample top --batch 64 --method nn --save_translation hinge.trans --only_test_translation --hinge_training_loss hinge_loss.npy" -o hinge.out;

sbatch -p TitanXx8 --gres=gpu:1 --wrap "python ../../bilingual_lexicon_induction.py $data_root/wiki-news-300d-1M.vec $data_root/crawl-300d-2M.vec --src_vocab_file $data_root/$split_dir/vocab --tgt_vocab_file $data_root/$split_dir/vocab -d $data_root/$split_dir/S_10K/all.seed -t $data_root/$split_dir/??.test --loss_type l2 --orth --method nn --save_translation l2.trans --only_test_translation" -o l2.out

sbatch -p TitanXx8 --gres=gpu:1 --wrap "python ../../bilingual_lexicon_induction.py $data_root/wiki-news-300d-1M.vec $data_root/crawl-300d-2M.vec --src_vocab_file $data_root/$split_dir/vocab --tgt_vocab_file $data_root/$split_dir/vocab -d $data_root/$split_dir/S_10K/all.seed -t $data_root/$split_dir/??.test --loss_type l2 --orth --method hnn --save_translation l2_hnn.trans --only_test_translation" -o l2_hnn.out
