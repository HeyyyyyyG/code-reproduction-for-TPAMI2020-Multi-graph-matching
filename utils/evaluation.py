import numpy as np
from statistics import mean


def evaluate_matching_accuracy(mat, mat_gt, num_graph, num_inlier):
    """
    :param mat: matching results, (num_graph, num_graph, num_node, num_node)
    :param mat_gt: matching ground truth, (num_graph, num_graph, num_node, num_node)
    :param num_graph: number of graphs, int
    :param num_inlier: number of inliers, int
    :return: accuracy
    """
    acc_list = []
    for i in range(num_graph):
        for j in range(num_graph):
            x = mat[i, j][:num_inlier, :num_inlier]
            x_gt = mat_gt[i, j][:num_inlier, :num_inlier]
            acc = np.sum(np.sum(np.abs(x - x_gt), 1) == 0) / num_inlier
            acc_list.append(acc)
    return mean(acc_list)
