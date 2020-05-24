import time
import os
import torch
import pickle
import numpy as np
from statistics import mean
from src.mgm_floyd import mgm_floyd
from src.rrwm import RRWM
from utils.data_prepare import DataGenerator
from utils.evaluation import evaluate_matching_accuracy
from utils.hungarian import hungarian
from utils.cfg import cfg as CFG

# set dataset and class for offline multi-graph matching test
dataset_name = "WILLOW-ObjectClass"
# you can remove some classes type during the debug process
class_name = ["Car", "Motorbike", "Face", "Winebottle", "Duck"]

# set parameters for offline multi-graph matching test
test_iter = 5  # test iteration for each class, please set it less than 5 or new data will be generated

# number of graphs, inliers and outliers only affect the generated data (when test_iter is larger than 5),
# these parameters will not be used when the data is loaded from TestPrepare.
test_num_graph = 24  # number of graphs in each test
test_num_inlier = 10  # number of inliers in each graph
test_num_outlier = 2  # number of outliers in each graph

rrwm = RRWM()
cfg = CFG()

print("Test begin: test online multi-graph matching on {}".format(dataset_name))
for i_test in range(len(class_name)):
    print("**************************************************************************")
    print("Test online multi-graph matching on class {}".format(class_name[i_test]))
    time_cost = []
    accuracy = []
    for i_iter in range(test_iter):
        # prepare affinity matrix data for graph matching

        # set the path for loading data
        test_data_path = "data" + "/" + "TestPrepare" + "/" + class_name[i_test] + "/" + "test_data_" + str(i_iter)
        if os.path.exists(test_data_path):
            # load data from "/TestPrepare/{ClassType}/test_data_{i_iter}"
            with open(test_data_path, "rb") as f:
                data = pickle.load(f)
        else:
            # if nothing can be loaded, generate new data and save it
            class_path = "data" + "/" + dataset_name + "/" + class_name[i_test]
            data = DataGenerator(
                data_path=class_path,
                num_graphs=test_num_graph,
                num_inlier=test_num_inlier,
                num_outlier=test_num_outlier
            )
            with open(test_data_path, "wb") as f:
                pickle.dump(data, f)

        # pairwise matching: RRWM

        # set the path for loading pairwise matching results
        tic = time.time()
        init_mat_path = "data" + "/" + "TestPrepare" + "/" + class_name[i_test] + "/" + "init_mat_" + str(i_iter)
        if os.path.exists(init_mat_path):
            # load pairwise matching results from "/TestPrepare/{ClassType}/init_mat_{i_iter}"
            with open(init_mat_path, "rb") as f:
                X = pickle.load(f)
        else:
            # if nothing can be loaded, generate the initial matching results and save them
            m, n = data.num_graphs, data.num_nodes
            Kt = torch.tensor(data.K).reshape(-1, n * n, n * n).cuda()
            ns_src = torch.ones(m * m).int().cuda() * n
            ns_tgt = torch.ones(m * m).int().cuda() * n
            X_continue = rrwm(Kt, n, ns_src, ns_tgt).reshape(m * m, n, n).transpose(1, 2).contiguous()
            X = hungarian(X_continue, ns_src, ns_tgt).reshape(m, m, n, n).cpu().detach().numpy()
            with open(init_mat_path, "wb") as f:
                pickle.dump(X, f)
        toc = time.time()

        # evaluate the initial matching results
        rrwm_acc = evaluate_matching_accuracy(X, data.gt, data.num_graphs, data.num_inlier)
        rrwm_time = toc - tic

        # apply MGM-Floyd to get better matching results
        tic = time.time()
        mat = mgm_floyd(X, data.K, data.num_graphs, data.num_nodes)
        toc = time.time()

        # evaluate the final matching results
        time_cost.append(toc - tic)
        acc = evaluate_matching_accuracy(mat, data.gt, data.num_graphs, data.num_inlier)
        accuracy.append(acc)

        print("iter: {}/{}, init_acc: {:.4f}, floyd_acc: {:.4f}, init_time: {:.4f}, floyd_time: {:.4f}".
              format(i_iter, test_iter, rrwm_acc, acc, rrwm_time, rrwm_time + toc - tic))
    avg_time = mean(time_cost)
    avg_acc = mean(accuracy)
    print("Performance on class {}, accuracy: {:.4f}, time: {:.4f}".format(class_name[i_test], avg_acc, avg_time))
    assert cfg.offline_acc[i_test] - 0.02 <= avg_acc and cfg.offline_time[i_test] + 0.3 >= avg_time, \
        "Test {} failed. {} accuracy required, but got {}; time cost should be less than {}, but got {}". \
            format(class_name[i_test], cfg.offline_acc[i_test] - 0.02, avg_acc, cfg.offline_time[i_test] + 0.3,
                   avg_time)
    print("Test {} passed".format(class_name[i_test]))
    print("")
