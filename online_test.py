import time
import torch
import os
import pickle
import numpy as np
from statistics import mean
from src.mgm_floyd import mgm_floyd
from src.mgm_spfa import mgm_spfa
from src.fast_spfa import fast_spfa
from src.rrwm import RRWM
from utils.data_prepare import DataGenerator
from utils.evaluation import evaluate_matching_accuracy
from utils.hungarian import hungarian
from utils.cfg import cfg as CFG

# set dataset and class for offline multi-graph matching test
dataset_name = "WILLOW-ObjectClass"
# you can remove some classes type during the debug process
class_name = ["Car", "Motorbike", "Face", "Winebottle", "Duck"]
# class_name = ["Car"]

# set parameters for offline multi-graph matching test
test_iter = 5  # test iteration for each class, please set it less than 5 or new data will be generated
test_mode = "mgm-spfa"  # mgm-spfa or fast-spaf

# number of graphs, inliers and outliers only affect the generated data (when test_iter is larger than 5),
# these parameters will not be used when the data is loaded from TestPrepare.
test_num_graph = 24  # number of graphs in each test
test_num_inlier = 10  # number of inliers in each graph
test_num_outlier = 2  # number of outliers in each graph

rrwm = RRWM()
cfg = CFG()

print("Test begin: test offline multi-graph matching on {}".format(dataset_name))
for i_test in range(len(class_name)):
    print("********************************************************************")
    print("Test on class {}".format(class_name[i_test]))
    online_acc = np.zeros((test_iter, test_num_graph))
    online_time = np.zeros((test_iter, test_num_graph))
    for i_iter in range(test_iter):
        # prepare affinity matrix data for graph matching

        # set the path for loading data
        print("iter: {}/{}".format(i_iter, test_iter))
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
        init_mat = X

        X = mgm_floyd(init_mat[:15, :15], data.K[:15, :15], 15, data.num_nodes)
        online_acc[i_iter][15] = evaluate_matching_accuracy(X, data.gt[:15, :15], 15, data.num_inlier)

        # apply MGM-SPFA to get better matching results
        for n_graph in range(16, data.num_graphs):
            tic = time.time()
            X_new = init_mat[:n_graph, :n_graph, :, :]
            X_new[:-1, :-1, :, :] = X
            if test_mode == "fast-spfa":
                X = fast_spfa(data.K[:n_graph, :n_graph, :, :], X_new, n_graph, data.num_nodes)
            elif test_mode == "mgm-spfa":
                X = mgm_spfa(data.K[:n_graph, :n_graph, :, :], X_new, n_graph, data.num_nodes)
            toc = time.time()
            acc = evaluate_matching_accuracy(X, data.gt[:n_graph, :n_graph], n_graph, data.num_inlier)
            t = toc - tic
            online_acc[i_iter][n_graph] = acc
            online_time[i_iter][n_graph] = t
            print("number of graphs: {}, accuracy: {:.4f}, time: {:.4f}".format(n_graph + 1, acc, t))
        print("")

    avg_time = np.mean(online_time, 0)
    avg_acc = np.mean(online_acc, 0)
    if test_mode == "mgm-spfa":
        online_acc = cfg.online_acc
        online_time = cfg.online_time
        _t = 0.2
    else:
        assert test_mode == "fast-spfa", "Test mode wrong, should be one of {mgm-spfa, fast-spfa}"
        online_acc = cfg.online_fast_acc
        online_time = cfg.online_fast_time
        _t = 0.1
    print("Overall performance on class {}".format(class_name[i_test]))
    for n_graph in range(16, test_num_graph):
        print("number of graphs: {}, accuracy: {:.4f}, time: {:.4f}"
              .format(n_graph + 1, avg_acc[n_graph], avg_time[n_graph]))
        if n_graph == test_num_graph - 1 :
            assert online_acc[i_test] - 0.02 <= avg_acc[n_graph] and online_time[i_test] + _t >= avg_time[n_graph], \
                "Test {} failed. {} accuracy required, but got {}; time cost should be less than {}, but got {}". \
                format(class_name[i_test], online_acc[i_test] - 0.02, avg_acc[n_graph], online_time[i_test] + _t, avg_time[n_graph])
    print("Online Test, mode {}, class {} passed".format(test_mode, class_name[i_test]))
    print("")
