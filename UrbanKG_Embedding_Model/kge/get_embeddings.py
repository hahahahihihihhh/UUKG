import argparse
import json
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import torch
import torch.optim
import pandas as pd
import numpy as np
import sys
sys.path.append("..")
import models
import optimizers.regularizers as regularizers
from datasets.kg_dataset import KGDataset
from models import all_models

DATA_PATH = '../data'
# model_path = "embedding_model/CHI/AttE"
# model_path = "embedding_model/CHI/AttH"
# model_path = "embedding_model/CHI/ComplEx"
# model_path = "embedding_model/CHI/DisMult"
# model_path = "embedding_model/CHI/GIE"
# model_path = "embedding_model/CHI/MuRE"
# model_path = "embedding_model/CHI/RandomE"
# model_path = "embedding_model/CHI/RefE"
# model_path = "embedding_model/CHI/RefH"
# model_path = "embedding_model/CHI/RotatE"
# model_path = "embedding_model/CHI/RotE"
# model_path = "embedding_model/CHI/RotH"
# model_path = "embedding_model/CHI/TransE"
# model_path = "embedding_model/CHI/TransH"

model_path = "../ke_model/NYC/GIE"


KG_id_path_prefix, save_path_prefix = "./K-Emb/used_xxx_id2KG_id", "xxx_embeddings"
args = None

def init_parser(config):
    parser = argparse.ArgumentParser(
        description="Urban Knowledge Graph Embedding"
    )
    parser.add_argument(
        "--dataset", default=config['dataset'], choices=["NYC", "CHI"],
        help="Urban Knowledge Graph dataset"
    )
    parser.add_argument(
        "--model", default=config['model'], choices=all_models,
        help='Model name: "TransE", "DisMult", "MuRE", "RotE", "RefE", "AttE", "ComplEx", "RotatE", "RotH", "RefH", "AttH", "TransH", "GIE"'
    )
    parser.add_argument(
        "--optimizer", default=config['optimizer'], choices=["Adagrad", "Adam", "SparseAdam"],
        help="Optimizer"
    )
    parser.add_argument(
        "--max_epochs", default=config['max_epochs'], type=int, help="Maximum number of epochs to train for"
    )
    parser.add_argument(
        "--patience", default=config['patience'], type=int, help="Number of epochs before early stopping"
    )
    parser.add_argument(
        "--valid", default=config['valid'], type=float, help="Number of epochs before validation"
    )
    parser.add_argument(
        "--rank", default=config['rank'], type=int, help="Embedding dimension"
    )
    parser.add_argument(
        "--batch_size", default=config['batch_size'], type=int, help="Batch size"   # 4096
    )
    parser.add_argument(
        "--learning_rate", default=config['learning_rate'], type=float, help="Learning rate"
    )
    parser.add_argument(
        "--neg_sample_size", default=config['neg_sample_size'], type=int,
        help="Negative sample size, -1 to not use negative sampling"
    )
    parser.add_argument(
        "--init_size", default=config['init_size'], type=float, help="Initial embeddings' scale"
    )
    parser.add_argument(
        "--multi_c", action="store_true", default=config['multi_c'], help="Multiple curvatures per relation"  # default = False
    )
    parser.add_argument(
        "--regularizer", choices=["N3", "F2"], default=config['regularizer'], help="Regularizer"
    )
    parser.add_argument(
        "--reg", default=config['reg'], type=float, help="Regularization weight"
    )
    parser.add_argument(
        "--dropout", default=config['dropout'], type=float, help="Dropout rate"
    )
    parser.add_argument(
        "--gamma", default=config['gamma'], type=float, help="Margin for distance-based losses"
    )
    parser.add_argument(
        "--bias", default=config['bias'], type=str, choices=["constant", "learn", "none"],
        help="Bias type (none for no bias)"
    )
    parser.add_argument(
        "--dtype", default=config['dtype'], type=str, choices=["single", "double"], help="Machine precision"
    )
    parser.add_argument(
        "--double_neg", action="store_true", default=config['double_neg'],
        help="Whether to negative sample both head and tail entities"
    )   # default = False
    parser.add_argument(
        "--debug", action="store_true", default=config['debug'],
        help="Only use 1000 examples for debugging"
    )   # default = False
    return parser

def get_embeddings(model_path):
    # load model
    dataset_path = os.path.join(DATA_PATH, args.dataset)
    dataset = KGDataset(dataset_path, args.debug)
    args.sizes = dataset.get_shape()
    print(args)
    model = getattr(models, args.model)(args)   # model initial
    model.load_state_dict(torch.load(
        # os.path.join("../logs/XXX/",
        #              "model.pt")))
        os.path.join(model_path, "model.pt")))   # fill parameters
    # get embeddings
    entity_embeddings = model.entity.weight.detach().numpy()
    rel_embeddings = model.rel.weight.detach().numpy()
    idx = pd.read_csv(os.path.join(DATA_PATH, args.dataset, "entities_idx.csv"), header=None)
    entity_idx = np.array(idx)
    entity_final_embedddings = np.zeros([entity_embeddings.shape[0], entity_embeddings.shape[1]])
    for _i in range(entity_embeddings.shape[0]):
        assert _i == int(entity_idx[_i])
        entity_final_embedddings[int(entity_idx[_i])] = entity_embeddings[_i]
    return entity_final_embedddings, rel_embeddings

def get_area_embeddings(area_id2KG_id_path, entity_final_embedddings, embedding_name):
    area = pd.read_csv(area_id2KG_id_path)
    area_id2KG_id = area[["area_id", "KG_id"]].values
    area_embeddings = []
    for _ in area_id2KG_id:
        area_embeddings.append(entity_final_embedddings[_[1]])
    save_path = os.path.join(save_path_prefix, args.dataset, args.model)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    np.save(os.path.join(save_path, embedding_name), area_embeddings)

def get_road_embeddings(road_id2KG_id_path, entity_final_embedddings, embedding_name):
    road = pd.read_csv(road_id2KG_id_path)
    road_id2KG_id = road[["road_id", "KG_id"]].values
    road_embeddings = []
    for _ in road_id2KG_id:
        road_embeddings.append(entity_final_embedddings[_[1]])
    save_path = os.path.join(save_path_prefix, args.dataset, args.model)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    np.save(os.path.join(save_path, embedding_name), road_embeddings)

def get_POI_embeddings(POI_id2KG_id_path, entity_final_embedddings, embedding_name):
    POI = pd.read_csv(POI_id2KG_id_path)
    POI_id2KG_id = POI[["POI_id", "KG_id"]].values
    POI_embeddings = []
    for _ in POI_id2KG_id:
        POI_embeddings.append(entity_final_embedddings[_[1]])
    save_path = os.path.join(save_path_prefix, args.dataset, args.model)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    np.save(os.path.join(save_path, embedding_name), POI_embeddings)

def get_rel_embeddings(rel_final_embedddings, embedding_name):
    save_path = os.path.join(save_path_prefix, args.dataset, args.model)
    nr = rel_final_embedddings.shape[0]
    np.save(os.path.join(save_path, embedding_name), rel_final_embedddings[:nr//2])

if __name__ == "__main__":
    # entity_final_embedddings, pca_final_embeddings, tsne_dim2_embeddings = get_embeddings(parser.parse_args())
    # load config file
    with open(os.path.join(model_path, "config.json")) as config:
        config = json.load(config)
        parser = init_parser(config)
        args = parser.parse_args()
    entity_final_embedddings, rel_final_embeddings = get_embeddings(model_path)
    get_area_embeddings(os.path.join(KG_id_path_prefix, args.dataset, "used_area_id2KG_id.csv"),
                        entity_final_embedddings,"area_{}d.npy".format(args.rank))
    get_road_embeddings(os.path.join(KG_id_path_prefix, args.dataset, "used_road_id2KG_id.csv"),
                        entity_final_embedddings,"road_{}d.npy".format(args.rank))
    get_POI_embeddings(os.path.join(KG_id_path_prefix, args.dataset, "used_POI_id2KG_id.csv"),
                        entity_final_embedddings,  "POI_{}d.npy".format(args.rank))
    get_rel_embeddings(rel_final_embeddings, "rel_{}d.npy".format(args.rank))















