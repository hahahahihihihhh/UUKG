import argparse
import json
import logging
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch
import torch.optim
import models
import optimizers.regularizers as regularizers
from datasets.kg_dataset import KGDataset
from models import all_models
from optimizers.kg_optimizer import KGOptimizer
from utils.train import get_savedir, avg_both, format_metrics, count_params

DATA_PATH = './data'
config = './config/config_CP.json'

def init_parser(config):
    parser = argparse.ArgumentParser(
        description="Urban Knowledge Graph Embedding"
    )
    parser.add_argument(
        "--dataset", default=config['dataset'], choices=["NYC", "CHI"],
        help="Urban Knowledge Graph dataset"
    )
    parser.add_argument(
        "--model", default=config['model'], choices=all_models, help='Model name'
    )
    parser.add_argument(
        "--optimizer", choices=["Adagrad", "Adam", "SparseAdam"], default=config['optimizer'],
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
        "--batch_size", default=config['batch_size'], type=int, help="Batch size"   # 4120
    )
    parser.add_argument(
        "--learning_rate", default=config['learning_rate'], type=float, help="Learning rate"
    )
    parser.add_argument(
        "--neg_sample_size", default=config['neg_sample_size'], type=int, help="Negative sample size, -1 to not use negative sampling"
    )
    parser.add_argument(
        "--init_size", default=config['init_size'], type=float, help="Initial embeddings' scale"
    )
    parser.add_argument(
        "--multi_c", action="store_true", default = config['multi_c'], help="Multiple curvatures per relation"    # default = False
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
    )   # default = false
    parser.add_argument(
        "--debug", action="store_true", default=config['debug'],
        help="Only use 1000 examples for debugging"
    )   # default = false
    return parser

def train(args):
    save_dir = get_savedir(args.model, args.dataset)

    # file logger
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=os.path.join(save_dir, "train.log")
    )

    # stdout logger
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s")
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)
    logging.info("Saving logs in: {}".format(save_dir))

    # create dataset
    dataset_path = os.path.join(DATA_PATH, args.dataset)
    dataset = KGDataset(dataset_path, args.debug)
    args.sizes = dataset.get_shape()

    # load data
    logging.info("\t " + str(dataset.get_shape()))
    train_examples = dataset.get_examples("train")
    valid_examples = dataset.get_examples("valid")
    test_examples = dataset.get_examples("test")
    filters = dataset.get_filters()

    # save config
    with open(os.path.join(save_dir, "config.json"), "w") as fjson:
        json.dump(vars(args), fjson)

    # create model
    model = getattr(models, args.model)(args)
    print(model)
    total = count_params(model)
    logging.info("Total number of parameters {}".format(total))
    device = "cuda"
    model.to(device)

    # get optimizer
    regularizer = getattr(regularizers, args.regularizer)(args.reg)
    optim_method = getattr(torch.optim, args.optimizer)(model.parameters(), lr=args.learning_rate)
    optimizer = KGOptimizer(model, regularizer, optim_method, args.batch_size, args.neg_sample_size,
                            bool(args.double_neg))
    counter = 0
    best_mrr = None
    best_epoch = None
    logging.info("\t Start training")

    # args.max_epochs
    for _step in range(args.max_epochs):

        # Train step
        model.train()
        train_loss = optimizer.epoch(train_examples)
        logging.info("\t Epoch {} | average train loss: {:.4f}".format(_step, train_loss))
        # Valid step
        model.eval()
        valid_loss = optimizer.calculate_valid_loss(valid_examples)
        logging.info("\t Epoch {} | average valid loss: {:.4f}".format(_step, valid_loss))

        if (_step + 1) % args.valid == 0:
            valid_metrics = avg_both(*model.compute_metrics(valid_examples, filters))
            logging.info(format_metrics(valid_metrics, split="valid"))

            valid_mrr = valid_metrics["MRR"]
            if not best_mrr or valid_mrr > best_mrr:
                best_mrr = valid_mrr
                counter = 0
                best_epoch = _step
                logging.info("\t Saving model at epoch {} in {}".format(_step, save_dir))
                torch.save(model.cpu().state_dict(), os.path.join(save_dir, "model.pt"))
                model.cuda()
            else:
                counter += 1
                if counter == args.patience:
                    logging.info("\t Early stopping")
                    break
                elif counter == args.patience // 2:
                    pass
    logging.info("\t Optimization finished")
    if not best_mrr:
        torch.save(model.cpu().state_dict(), os.path.join(save_dir, "model.pt"))
    else:
        logging.info("\t Loading best model saved at epoch {}".format(best_epoch))
        model.load_state_dict(torch.load(os.path.join(save_dir, "model.pt")))
    model.cuda()
    model.eval()

    # Validation metrics
    valid_metrics = avg_both(*model.compute_metrics(valid_examples, filters))
    logging.info(format_metrics(valid_metrics, split="valid"))

    # Test metrics
    test_metrics = avg_both(*model.compute_metrics(test_examples, filters))
    logging.info(format_metrics(test_metrics, split="test"))

if __name__ == "__main__":
    with open(config) as config:
        config = json.load(config)
        print(config)
        parser = init_parser(config)
        train(parser.parse_args())
