"""
训练并评估单一模型的脚本
"""
import argparse
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'
from libcity.pipeline import run_model
from libcity.utils import str2bool, add_general_args
import random
import pandas as pd
import numpy as np

log = './logs/'
cache = './libcity/cache/'

if __name__ == '__main__':
    total = 5
    exp_ids, modelName, datasetName = [], "", ""
    hasImproved = False
    for i in range(total):
        seed = random.randint(0, 10000)

        parser = argparse.ArgumentParser()
        # 增加指定的参数
        parser.add_argument('--task', type=str,
                            default='traffic_state_pred', help='the name of task')
        parser.add_argument('--model', type=str,
                            default='AGCRN', help='the name of model')
        # CHITaxi20190406
        parser.add_argument('--dataset', type=str,
                            default='CHITaxi20190406', help='the name of dataset')
        parser.add_argument('--config_file', type=str,
                            default=None, help='the file name of config file')
        parser.add_argument('--saved_model', type=str2bool,
                            default=True, help='whether save the trained model')
        parser.add_argument('--train', type=str2bool, default=True,
                            help='whether re-train model if the model is trained before')
        parser.add_argument('--exp_id', type=str, default=str(seed), help='id of experiment')
        parser.add_argument('--seed', type=int, default=seed, help='random seed')
        parser.add_argument('--load_external', type=str2bool, default=True, help="whether to load external data")
        # 增加其他可选的参数
        add_general_args(parser)
        # 解析参数
        args = parser.parse_args()
        dict_args = vars(args)
        hasImproved = dict_args['load_external']
        exp_ids.append(dict_args['exp_id'])
        modelName, datasetName = dict_args['model'], dict_args['dataset']
        other_args = {key: val for key, val in dict_args.items() if key not in [
            'task', 'model', 'dataset', 'config_file', 'saved_model', 'train'] and
            val is not None}
        run_model(task=args.task, model_name=args.model, dataset_name=args.dataset,
                  config_file=args.config_file, saved_model=args.saved_model,
                  train=args.train, other_args=other_args)
    print("improved: ", hasImproved)
    print(exp_ids)

    save_path = log + datasetName[0:3] + '/' + modelName + '_' + datasetName + ('_improved' if hasImproved else '_normal') + '/'
    final_results_ten_train = np.zeros([12, 2, total])

    eval_metrics = ["MAE", "RMSE"]
    for i in range(len(exp_ids)):
        exp_id = exp_ids[i]
        result = pd.read_csv(cache + exp_id + '/' + 'evaluate_cache/' + modelName + '_' + datasetName + '.csv')
        temp = result[eval_metrics].values
        final_results_ten_train[:, :, i] = temp

    avg_results = np.zeros([12, 2])
    std_results = np.zeros([12, 2])
    ## 计算 指标均值 和 标准差
    for j in range(final_results_ten_train.shape[0]):
        avg_mae_rmse = np.mean(final_results_ten_train[j, :, :], axis=1)
        std_mae = np.std(final_results_ten_train[j, :, :][0])
        std_rmse = np.std(final_results_ten_train[j, :, :][1])

        avg_results[j] = avg_mae_rmse
        std_results[j][0] = std_mae
        std_results[j][1] = std_rmse

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with open(save_path + 'experiment_ids.txt', 'w') as f:
        f.write(', '.join(exp_ids))

    print(avg_results, std_results)
    pd.DataFrame(avg_results, columns=eval_metrics).to_csv(save_path + 'avg_result.csv', index=False)
    pd.DataFrame(std_results, columns=eval_metrics).to_csv(save_path + 'std_result.csv', index=False)

    avg_steps_results = np.mean(avg_results, axis = 0)
    std_steps_results = np.mean(std_results, axis = 0)
    print(avg_steps_results, std_steps_results)

    pd.DataFrame(np.array(avg_steps_results).reshape(1, 2), columns=eval_metrics).to_csv(save_path + 'avg_steps_result.csv', index=False)
    pd.DataFrame(np.array(std_steps_results).reshape(1, 2), columns=eval_metrics).to_csv(save_path + 'std_steps_result.csv', index=False)


# ['9605', '2943', '8415', '8815', '2476']  AGCRN
# ['4203', '7774', '1921', '168', '1882']   TranE + AGCRN