#!/bin/sh
#python pipeline.py --dataset_name bbbp --random_split True
#python pipeline.py --dataset_name mutag --model_name gin \

#python pipeline.py --dataset_name BA_2Motifs --random_split True \
#        --latent_dim 20 20 20 --adj_normlize False --emb_normlize True\
#
#python pipeline.py --dataset_name Graph_SST2 --model_name gat \
#        --random_split False

python pipeline.py --dataset_name BA_shapes --random_split True \
      --latent_dim 20 20 20 --concate True --adj_normlize False --emb_normlize True