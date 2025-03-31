
# Default
python data_split.py


# python data_split.py \
#     --r2_input_path integrated_data/viridiplantae_dataset_partitioned \
#     --r2_train_path integrated_data/train_split_parquet \
#     --r2_test_path integrated_data/test_split_parquet


# Using DBSCAN with Custom Parameters
# python data_split.py \
#     --cluster_method dbscan \
#     --eps 0.4 \
#     --min_samples 10 \
#     --r2_train_path integrated_data/train_dbscan_eps04_min10_parquet \
#     --r2_test_path integrated_data/test_dbscan_eps04_min10_parquet \
#     --test_ratio 0.25

# Using Agglomerative Clustering with Distance Threshold
# python data_split.py \
#     --cluster_method agglomerative \
#     --distance_threshold 0.7 \
#     --r2_train_path integrated_data/train_agg_dist07_parquet \
#     --r2_test_path integrated_data/test_agg_dist07_parquet


# Using Agglomerative Clustering with Fixed Number of Clusters
# python data_split.py \
#     --cluster_method agglomerative \
#     --n_clusters 100 \
#     --r2_train_path integrated_data/train_agg_k100_parquet \
#     --r2_test_path integrated_data/test_agg_k100_parquet


# Using a Smaller/Faster Model and Saving Embeddings
# python data_split.py \
#     --esm_model facebook/esm2_t12_35M_UR50D \
#     --batch_size 64 \
#     --cluster_method dbscan \
#     --eps 0.35 \
#     --min_samples 5 \
#     --embeddings_out embeddings_35M.npy \
#     --ids_out embedding_ids_35M.txt \
#     --r2_train_path integrated_data/train_35M_dbscan_parquet \
#     --r2_test_path integrated_data/test_35M_dbscan_parquet