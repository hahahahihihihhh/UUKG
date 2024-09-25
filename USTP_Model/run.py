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
ke_method = 'Concat'
config_file = 'config/CHITaxi20190406/GWNET/{}/config_GIE.json'.format(ke_method)

def train(config, total = 5):
    predict_steps, eval_metrics = config['output_window'], config["metrics"]
    # extTime = config.get('load_external', False) and (config.get("add_time_in_day", False) or config.get("add_day_in_week", False))
    # extSpace = config.get('load_external', False) and config.get('ke_model', None)
    # if extTime and extSpace:
    #     ext = "ExtTime&Space"
    # elif extTime:
    #     ext = 'ExtTime'
    # elif extSpace:
    #     ext = 'ExtSpace'
    # else:
    #     ext = 'ExtNone'
    # emb = ''
    # ext, ke_model = 'ExtNone', ''
    # if extSpace:
    #     ext, ke_model = "ExtSpace", config.get('ke_model', '')

    ke_model = config.get('ke_model', '')
    save_dir = os.path.join(log, config['dataset'], config['model'], ke_method, ke_model)

    # save_dir = os.path.join(log, config['dataset'], config['model'], 'Test')
    ensure_dir(save_dir)
    # print(save_dir)

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
                     .format(config.get('ke_model', 'None'), exp_ids))
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
