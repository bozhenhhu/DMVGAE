import os

def run_func(description, ppi_path, pseq_path, vec_path,
            split_new, split_mode, train_valid_index_path,
            use_lr_scheduler, save_path, graph_only_train, 
            batch_size, epochs, use_dmt, v_input, v_latent, sigmaP, 
            augNearRate, balance, seed):
    os.system("python -u src/benchmark/GNN_PPI/gnn_train_dmt.py \
            --description={} \
            --ppi_path={} \
            --pseq_path={} \
            --vec_path={} \
            --split_new={} \
            --split_mode={} \
            --train_valid_index_path={} \
            --use_lr_scheduler={} \
            --save_path={} \
            --graph_only_train={} \
            --batch_size={} \
            --epochs={} \
            --use_dmt={} \
            --v_input={} \
            --v_latent={} \
            --sigmaP={} \
            --augNearRate={} \
            --balance={} \
            --seed={}".format(description, ppi_path, pseq_path, vec_path, 
                    split_new, split_mode, train_valid_index_path,
                    use_lr_scheduler, save_path, graph_only_train, 
                    batch_size, epochs, use_dmt, v_input, v_latent, sigmaP, augNearRate, balance, seed))

if __name__ == "__main__":
    description = "KeAP20_string"

#     ppi_path = './data/protein.actions.SHS27k.STRING.txt'
#     pseq_path = "data/protein.SHS27k.sequences.dictionary.tsv"
#     vec_path = "data/PPI_embeddings/protein_embedding_KeAP20_shs27k.npy"
#     ppi_path = "./data/protein.actions.SHS148k.STRING.txt"
#     pseq_path = "./data/protein.SHS148k.sequences.dictionary.tsv"
#     vec_path = "./data/PPI_embeddings/protein_embedding_KeAP20_shs148k.npy"
    ppi_path = 'data/9606.protein.actions.all_connected.txt'
    pseq_path = 'data/protein.STRING_all_connected.sequences.dictionary.tsv'
    vec_path = 'data/PPI_embeddings/protein_embedding_KeAP20_STRING.npy'

    split_new = "True"
    split_mode = "dfs"
#     train_valid_index_path = "./data/new_train_valid_index_json/STRING.dfs.fold1.json"
#     train_valid_index_path = "./data/new_train_valid_index_json/SHS148k.bfs.fold1.json"
    train_valid_index_path = "data/new_train_valid_index_json/STRING.{}.fold1.json".format(split_mode)

    use_lr_scheduler = "True"
    save_path = "./output/ppi"
    graph_only_train = "False"

    batch_size = 2048
    epochs = 300

    use_dmt = 'True'
    v_input = 100
    v_latent = 0.01
    sigmaP = 1.0
    augNearRate = 100000
    balance = 0.01
    seeds = [0]

    for seed in seeds:
        run_func(description, ppi_path, pseq_path, vec_path, 
                split_new, split_mode, train_valid_index_path,
                use_lr_scheduler, save_path, graph_only_train, 
                batch_size, epochs, use_dmt, v_input, v_latent, sigmaP, 
                augNearRate, balance, seed
                )