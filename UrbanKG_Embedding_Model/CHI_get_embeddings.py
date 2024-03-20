import argparse
import json
import logging
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import torch
import torch.optim
import pandas as pd
import numpy as np
import models
import optimizers.regularizers as regularizers
from datasets.kg_dataset import KGDataset
from models import all_models
DATA_PATH = './data'

parser = argparse.ArgumentParser(
    description="Urban Knowledge Graph Embedding"
)
parser.add_argument(
    "--dataset", default="CHI", choices=["NYC", "CHI"],
    help="Urban Knowledge Graph dataset"
)
parser.add_argument(
    "--model", default="TransE", choices=all_models, help='"TransE", "CP", "MurE", "RotE", "RefE", "AttE",'
                                                       '"ComplEx", "RotatE",'
                                                       '"RotH", "RefH", "AttH"'
                                                       '"GIE'
)
parser.add_argument(
    "--optimizer", choices=["Adagrad", "Adam", "SparseAdam"], default="Adam",
    help="Optimizer"
)
parser.add_argument(
    "--max_epochs", default=150, type=int, help="Maximum number of epochs to train for"
)
parser.add_argument(
    "--patience", default=10, type=int, help="Number of epochs before early stopping"
)
parser.add_argument(
    "--valid", default=3, type=float, help="Number of epochs before validation"
)
parser.add_argument(
    "--rank", default=32, type=int, help="Embedding dimension"
)
parser.add_argument(
    "--batch_size", default=4096, type=int, help="Batch size"   # 4096
)
parser.add_argument(
    "--learning_rate", default=1e-3, type=float, help="Learning rate"
)
parser.add_argument(
    "--neg_sample_size", default=50, type=int, help="Negative sample size, -1 to not use negative sampling"
)
parser.add_argument(
    "--init_size", default=1e-3, type=float, help="Initial embeddings' scale"
)
parser.add_argument(
    "--multi_c", action="store_true", help="Multiple curvatures per relation"
)
parser.add_argument(
    "--regularizer", choices=["N3", "F2"], default="N3", help="Regularizer"
)
parser.add_argument(
    "--reg", default=0, type=float, help="Regularization weight"
)
parser.add_argument(
    "--dropout", default=0, type=float, help="Dropout rate"
)
parser.add_argument(
    "--gamma", default=0, type=float, help="Margin for distance-based losses"
)
parser.add_argument(
    "--bias", default="constant", type=str, choices=["constant", "learn", "none"],
    help="Bias type (none for no bias)"
)
parser.add_argument(
    "--dtype", default="double", type=str, choices=["single", "double"], help="Machine precision"
)
parser.add_argument(
    "--double_neg", action="store_true",
    help="Whether to negative sample both head and tail entities"
)
parser.add_argument(
    "--debug", action="store_true",
    help="Only use 1000 examples for debugging"
)

model_path = "01_31/CHI/TransE_12_05_03/"
def get_embeddings(args):
    # create model
    dataset_path = os.path.join(DATA_PATH, args.dataset)
    dataset = KGDataset(dataset_path, args.debug)
    args.sizes = dataset.get_shape()

    model = getattr(models, args.model)(args)
    model.load_state_dict(torch.load(
        # os.path.join("../logs/XXX/",
        #              "model.pt")))
        os.path.join("./logs/" + model_path,
                     "model.pt")))
    entity_embeddings = model.entity.weight.detach().numpy()
    idx = pd.read_csv(DATA_PATH + '/' + args.dataset + "/entity_idx.csv", header=None)
    entity_idx = np.array(idx)
    entity_final_embedddings = np.zeros([entity_embeddings.shape[0], entity_embeddings.shape[1]])
    for i in range(entity_embeddings.shape[0]):
        entity_final_embedddings[int(entity_idx[i])] = entity_embeddings[i]
    return entity_final_embedddings


def get_area_embeddings(area_KG_id_path, entity_final_embedddings, save_path):
    area = pd.read_csv(area_KG_id_path)
    area_KG_id = area[["area_id", "KG"]].values
    area_embeddings = []
    for i in range(area_KG_id.shape[0]):
        area_embeddings.append(entity_final_embedddings[int(area_KG_id[i][1])])
    np.save(save_path, np.array(area_embeddings))


def get_road_embeddings(road_KG_id_path, entity_final_embedddings, save_path):
    road = pd.read_csv(road_KG_id_path)
    road_KG_id = road[["road_id", "KG"]].values
    road_embeddings = []
    for i in range(road_KG_id.shape[0]):
        road_embeddings.append(entity_final_embedddings[int(road_KG_id[i][1])])
    np.save(save_path, np.array(road_embeddings))

def get_POI_embeddings(POI_KG_id_path, entity_final_embedddings, save_path):
    POI = pd.read_csv(POI_KG_id_path)
    POI_KG_id = POI[["POI_id", "KG"]].values
    POI_embeddings = []
    for i in range(POI_KG_id.shape[0]):
        POI_embeddings.append(entity_final_embedddings[int(POI_KG_id[i][1])])
    np.save(save_path, np.array(POI_embeddings))

# entity_final_embedddings, pca_final_embeddings, tsne_dim2_embeddings = get_embeddings(parser.parse_args())
entity_final_embedddings = get_embeddings(parser.parse_args())

KG_id_path_prefix, save_path_prefix = "KG/used_xxx_KG_id/", "KG/xxx_embeddings/"

get_area_embeddings(KG_id_path_prefix + "CHI_used_area_KG_id.csv", entity_final_embedddings,
                      save_path_prefix + "CHI_area_embeddings.npy")

get_road_embeddings(KG_id_path_prefix + "CHI_used_road_KG_id.csv", entity_final_embedddings,
                      save_path_prefix + "CHI_road_embeddings.npy")

get_POI_embeddings(KG_id_path_prefix + "CHI_used_POI_KG_id.csv", entity_final_embedddings,
                      save_path_prefix + "CHI_POI_embeddings.npy")















