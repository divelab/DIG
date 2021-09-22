#!/bin/sh
#python -m forgraph.subgraphx --dataset_name bbbp --random_split True --reward_method mc_l_shapley
#python -m forgraph.subgraphx --dataset_name mutag --model_name gin \
#        --c_puct 10.0 --reward_method mc_l_shapley
#
#python -m forgraph.subgraphx --dataset_name BA_2Motifs --random_split True \
#        --latent_dim 20 20 20 --adj_normlize False --emb_normlize True \
#        --readout mean --c_puct 10.0 --min_atom 5 --reward_method mc_l_shapley

#python -m forgraph.subgraphx --dataset_name grt_sst2_BERT_Identity --model_name gat \
#        --random_split False --reward_method mc_l_shapley

python -m fornode.subgraphx --dataset_name BA_shapes --random_split True \
      --latent_dim 20 20 20 --concate True --adj_normlize False --emb_normlize True \
      --high2low True --min_atom 5 --reward_method nc_mc_l_shapley
