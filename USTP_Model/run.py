"""
训练并评估单一模型的脚本
"""
import argparse
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'
from libcity.pipeline import run_model
from libcity.utils import ensure_dir
import pandas as pd
import numpy as np
import logging
import json

log = './logs/'
cache = './libcity/cache/'
config_file = 'config/CHI/AGCRN/config_GIE.json'

# def init_parser(config):
#     seed = random.randint(0, 10000)
#     parser = argparse.ArgumentParser()
#     # 增加指定的参数
#     parser.add_argument('--task', type=str,
#                         default=config['task'], help='the name of task')
#     parser.add_argument('--model', type=str,
#                         default=config['model'], help='the name of model')
#     parser.add_argument('--dataset', type=str,
#                         default=config['dataset'], help='the name of dataset')
#     parser.add_argument('--config_file', type=str,
#                         default=config_file, help='the file name of config file')
#     # parser.add_argument('--saved_model', type=str2bool,
#     #                     default=config['saved_model'], help='whether save the trained model')
#     # parser.add_argument('--train', type=str2bool, default=True,
#     #                     help='whether re-train model if the model is trained before')
#     parser.add_argument('--exp_id', type=str, default=str(seed), help='id of experiment')
#     parser.add_argument('--seed', type=int, default=seed, help='random seed')
#     # parser.add_argument('--load_external', type=str2bool, default=True, help="whether to load external data")
#     # 增加其他可选的参数
#     add_general_args(parser)
#     return parser

def train(config, total = 5):
    predict_steps, eval_metrics = config['output_window'], config["metrics"]
    extTime = config.get('load_external', False) and (config.get("add_time_in_day", False) or config.get("add_day_in_week", False))
    extSpace = config.get('load_external', False) and config.get('embedding_model', None)
    if extTime and extSpace:
        ext = "ExtTime&Space"
    elif extTime:
        ext = 'ExtTime'
    elif extSpace:
        ext = 'ExtSpace'
    else:
        ext = 'ExtNone'

    if extSpace:
        scl = 'Scaler/'
        if config.get('normal_external', False):
            scl += config.get('ext_scaler', 'none')
        else:
            scl += 'none'
    else:
        scl = ''

    if extSpace:
        emb = config.get('embedding_model', 'NoneE')
    else:
        emb = ''

    save_dir = os.path.join(log, config['dataset'], config['model'],
                            ext,
                            scl,
                            emb)
    print(save_dir)

    # save_dir = os.path.join(log, config['dataset'], config['model'], 'Test')
    ensure_dir(save_dir)

    # file logger
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=os.path.join(save_dir, "train.log"),
        filemode='w'
    )

    # stdout logger
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s")
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)
    logging.info("Saving logs in: {}".format(save_dir))

    # create logger
    logger = logging.getLogger()

    # 初始化
    exp_ids = []
    final_results_five_train = np.zeros([predict_steps, len(eval_metrics), total])

    # 独立训练total次
    for _i in range(total):
        # 解析参数
        config['cur_times'] = _i
        print(config)
        exp_ids.append(config.get('exp_id', 0) + config.get('cur_times', 0))
        other_args = {key: val for key, val in config.items() if key not in [
            'task', 'model', 'dataset', 'config_file', 'saved_model', 'train'] and
            val is not None}
        # 运行模型, 获取指标
        result = run_model(task=config['task'], model_name=config['model'], dataset_name=config['dataset'],
                 other_args=other_args)
        # 获取指标
        # result = pd.read_csv(cache + str(config['exp_id']) + '/' + 'evaluate_cache/' + modelName + '_' + datasetName + '.csv')
        final_results_five_train[:, :, _i] = result[eval_metrics].values

    logger.info('----------------------------------------------------------------------------')
    logger.info("Kownledge Graph Embedding: {}, Experiment ids: {}"
                 .format(config.get('embedding_model', 'Normal'), exp_ids))
    # 计算指标均值和标准差
    avg_results = np.zeros([predict_steps, len(eval_metrics)])
    std_results = np.zeros([predict_steps, len(eval_metrics)])
    for _i in range(final_results_five_train.shape[0]):
        avg_mae_rmse = np.mean(final_results_five_train[_i, :, :], axis=1)
        std_mae = np.std(final_results_five_train[_i, :, :][0])
        std_rmse = np.std(final_results_five_train[_i, :, :][1])

        avg_results[_i] = avg_mae_rmse
        std_results[_i][0] = std_mae
        std_results[_i][1] = std_rmse

    # 保存各预测步指标均值和标准差
    df_avg_results = pd.DataFrame(avg_results, columns=eval_metrics)
    df_std_results = pd.DataFrame(std_results, columns=eval_metrics)
    df_avg_results.to_csv(os.path.join(save_dir, 'avg_result.csv'), index=False)
    df_std_results.to_csv(os.path.join(save_dir, 'std_result.csv'), index=False)
    logger.info('----------------------------------------------------------------------------')
    logger.info('Average: \n{} '.format(df_avg_results))
    logger.info('Standard deviation: \n{}'.format(df_std_results))

    # 保存所有预测步指标均值和标准差
    avg_steps_results = np.mean(avg_results, axis=0)
    std_steps_results = np.mean(std_results, axis=0)
    df_avg_steps_results = pd.DataFrame(np.array(avg_steps_results).reshape(1, 2), columns=eval_metrics)
    df_std_steps_results = pd.DataFrame(np.array(std_steps_results).reshape(1, 2), columns=eval_metrics)
    df_avg_steps_results.to_csv(os.path.join(save_dir, 'avg_steps_result.csv'), index=False)
    df_std_steps_results.to_csv(os.path.join(save_dir, 'std_steps_result.csv'), index=False)
    logger.info('----------------------------------------------------------------------------')
    logger.info('{} steps Average: \n{}'.format(predict_steps, df_avg_steps_results))
    logger.info('{} steps Standard deviation: \n{}'.format(predict_steps, df_std_steps_results))

if __name__ == '__main__':
    with open(config_file) as config:
        config = json.load(config)
        train(config)
