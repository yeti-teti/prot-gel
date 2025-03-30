
# Default
# python data_split.py --input ../../data/integrated_data.json

# Using DBSCAN with Custom Parameters
# python data_split.py \
#     --input ../../data/integrated_data.json \
#     --cluster_method dbscan \
#     --eps 0.4 \
#     --min_samples 10 \
#     --train_out train_dbscan_eps04_min10.json \
#     --test_out test_dbscan_eps04_min10.json \
#     --test_ratio 0.25

# Using Agglomerative Clustering with Distance Threshold
# python data_split.py \
#     --input integrated_data.json \
#     --cluster_method agglomerative \
#     --distance_threshold 0.7 \
#     --train_out train_agg_dist07.json \
#     --test_out test_agg_dist07.json

# Using Agglomerative Clustering with Fixed Number of Clusters
# python data_split.py \
#     --input integrated_data.json \
#     --cluster_method agglomerative \
#     --n_clusters 100 \
#     --train_out train_agg_k100.json \
#     --test_out test_agg_k100.json

# Using a Smaller/Faster Model and Saving Embeddings
python data_split.py \
    --input ../../data/integrated_data.json \
    --esm_model facebook/esm2_t12_35M_UR50D \
    --batch_size 64 \
    --cluster_method dbscan \
    --eps 0.35 \
    --min_samples 5 \
    --embeddings_out embeddings_35M.npy \
    --ids_out embedding_ids_35M.txt \
    --train_out ../../data/train.json \
    --test_out ../../data/test.json