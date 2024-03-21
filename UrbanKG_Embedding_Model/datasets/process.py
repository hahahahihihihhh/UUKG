"""Knowledge Graph dataset pre-processing functions."""

import collections
import os
import pickle
import numpy as np

DATA_PATH = "../data"

def get_idx(path, dataset_name):
    """Map entities and relations to unique ids.

    Args:
      path: path to directory with raw dataset files (tab-separated train/valid/test triples)

    Returns:
      ent2idx: Dictionary mapping raw entities to unique ids
      rel2idx: Dictionary mapping raw relations to unique ids
    """
    entities, relations = set(), set()
    for _split in ["train", "valid", "test"]:
        with open(os.path.join(path, _split + "_" + dataset_name + ".txt"), "r") as lines:
            for _line in lines:
                lhs, rel, rhs = _line.strip().split("\t")
                entities.add(int(lhs))
                entities.add(int(rhs))
                relations.add(int(rel))
    ent2idx = {x: i for (i, x) in enumerate(sorted(entities))}
    rel2idx = {x: i for (i, x) in enumerate(sorted(relations))}
    return ent2idx, rel2idx

def to_np_array(dataset_file, ent2idx, rel2idx):
    """Map raw dataset file to numpy array with unique ids.

    Args:
      dataset_file: Path to file containing raw triples in a split
      ent2idx: Dictionary mapping raw entities to unique ids
      rel2idx: Dictionary mapping raw relations to unique ids

    Returns:
      Numpy array of size n_examples x 3 mapping the raw dataset file to ids
    """
    examples = []
    with open(dataset_file, "r") as lines:
        for _line in lines:
            lhs, rel, rhs = _line.strip().split("\t")
            lhs = int(lhs); rel = int(rel); rhs = int(rhs)
            try:
                examples.append([ent2idx[lhs], rel2idx[rel], ent2idx[rhs]])
            except ValueError:
                continue
    return np.array(examples).astype("int64")

def get_filters(examples, n_relations):
    """Create filtering lists for evaluation.

    Args:
      examples: Numpy array of size n_examples x 3 containing KG triples
      n_relations: Int indicating the total number of relations in the KG

    Returns:
      lhs_final: Dictionary mapping queries (entity, relation) to filtered entities for left-hand-side prediction
      rhs_final: Dictionary mapping queries (entity, relation) to filtered entities for right-hand-side prediction
    """
    lhs_filters = collections.defaultdict(set)
    rhs_filters = collections.defaultdict(set)
    for _lhs, _rel, _rhs in examples:
        rhs_filters[(_lhs, _rel)].add(_rhs)
        lhs_filters[(_rhs, _rel + n_relations)].add(_lhs)          # to distinguish with the upper relation
    lhs_final = {}
    rhs_final = {}
    for _k, _v in lhs_filters.items():
        lhs_final[_k] = sorted(list(_v))
    for _k, _v in rhs_filters.items():
        rhs_final[_k] = sorted(list(_v))
    return lhs_final, rhs_final

def process_dataset(dataset_path, dataset_name):
    """Map entities and relations to ids and saves corresponding pickle arrays.

    Args:
      path: Path to dataset directory

    Returns:
      examples: Dictionary mapping splits to with Numpy array containing corresponding KG triples.
      filters: Dictionary containing filters for lhs and rhs predictions.
    """
    ent2idx, rel2idx = get_idx(dataset_path, dataset_name)
    entities_idx = list(ent2idx.keys())
    relations_idx = list(rel2idx.keys())

    # The index between UrbanKG id and embedding
    np.savetxt(dataset_path + "/entities_idx.csv", np.array(entities_idx), encoding="utf-8", delimiter=",")
    np.savetxt(dataset_path + "/relations_idx.csv", np.array(relations_idx), encoding="utf-8", delimiter=",")

    examples = {}
    splits = ["train", "valid", "test"]
    for _split in splits:
        dataset_file = os.path.join(dataset_path, _split + "_" + dataset_name + ".txt")
        examples[_split] = to_np_array(dataset_file, ent2idx, rel2idx)
    all_examples = np.concatenate([examples[_split] for _split in splits], axis=0)
    lhs_skip, rhs_skip = get_filters(all_examples, len(rel2idx))
    filters = {"lhs": lhs_skip, "rhs": rhs_skip}
    return examples, filters

if __name__ == "__main__":
    for _dataset_name in os.listdir(DATA_PATH):
        dataset_path = os.path.join(DATA_PATH, _dataset_name)
        dataset_examples, dataset_filters = process_dataset(dataset_path, _dataset_name)
        for _split in ["train", "valid", "test"]:
            with open(os.path.join(dataset_path, _split + ".pickle"), "wb") as save_file:
                pickle.dump(dataset_examples[_split], save_file)
        with open(os.path.join(dataset_path, "to_skip.pickle"), "wb") as save_file:
            pickle.dump(dataset_filters, save_file)
