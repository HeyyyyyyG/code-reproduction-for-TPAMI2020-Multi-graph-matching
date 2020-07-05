import torch
import numpy as np
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def mgm_floyd(X, K, num_graph, num_node):
    """
    :param K: affinity matrix, (num_graph, num_graph, num_node^2, num_node^2)
    :param num_graph: number of graph, int
    :param num_node: number of node, int
    :return: matching results, (num_graph, num_graph, num_node, num_node)
    """
    #print(X.shape,num_graph,num_node)

    lamb = 0.15
    beta= 0.15

    X = torch.Tensor(X).to(device)
    K = torch.Tensor(K).to(device)

    def cal_pairwise_consistency2(X):
        num_graph = X.size(0)
        num_node = X.size(2)
        #cp2=torch.ones([num_graph,num_graph]).to(device)
        cp3=torch.ones([num_graph,num_graph]).to(device)
        '''
        for i in range(X.size(0)):
            for j in range(X.size(1)):
                for k in range(X.size(0)):
                    for k2 in range(X.size(1)):
                        #print(torch.abs(torch.matmul(X[i][k],X[k][j])-torch.matmul(X[i][k2],X[k2][j])))
                        cp2[i][j] -= torch.abs(torch.matmul(X[i][k],X[k][j])-torch.matmul(X[i][k2],X[k2][j])).sum()/(2 * num_graph* (num_graph-1) * num_node)
        '''
        X_t = X.transpose(0,1)
        tmp = torch.matmul(X[:, None], X_t[None, ...])

        for k in range(X.size(0)):
            for k2 in range(X.size(1)):
                cp3 -= torch.abs(tmp[:,:,k]-tmp[:,:,k2]).sum((2,3))/(2 * num_graph* (num_graph-1) * num_node)

        return cp3
    def cal_pairwise_consistency(X):
        """
        :param X: matching results, (num_graph, num_graph, num_node, num_node) torch tensor
        :return: pairwise_consistency: (num_graph, num_graph) torch tensor
        """
        num_graph = X.size(0)
        num_node = X.size(2)
        X_t = X.transpose(0,1)
        pairwise_consistency = 1 - torch.abs(X[:, :, None] - torch.matmul(X[:, None], X_t[None, ...])).sum((2, 3, 4)) / (2 * num_graph * num_node)
        return pairwise_consistency


    def cal_affinity_score_for_all(X, K):
        #calculate affinity score for all pair of graphs
        # return: normalized_affinity_score: (num_graph, num_graph) torch tensor

        XT = X.transpose(2,3)# in order for column-wise
        vector_x = XT.reshape(num_graph, num_graph, -1, 1) #(num_graph, num_graph, num_node^2,1)
        vector_xT = vector_x.transpose(2,3)
        affinity_score = torch.matmul(torch.matmul(vector_xT, K), vector_x)  #(num_graph, num_graph, 1, 1)
        global X_norm
        X_norm = torch.max(affinity_score)
        normalized_affinity_score = affinity_score.reshape(num_graph, num_graph) / X_norm #(num_graph, num_graph)
        return normalized_affinity_score

    #diag = torch.eye(num_graph)<0.5
    #diag = diag.to(device)
    for k in range(num_graph):
        Xopt = torch.matmul(X[:, k, None], X[k, None, :])
        Sorg = cal_affinity_score_for_all(X, K)
        Sopt = cal_affinity_score_for_all(Xopt, K)
        update = (Sopt > Sorg)[:, :, None, None]

        update[np.diag_indices(num_graph)] = False

        X = update * Xopt + ~update * X

    for k in range(num_graph):
        cp = cal_pairwise_consistency(X)
        cp2 = cal_pairwise_consistency2(X)
        #print(cp2)
        Xopt = torch.matmul(X[:, k, None], X[k, None, :])
        Sorg = (1-lamb) * cal_affinity_score_for_all(X, K) + lamb * cp + beta * cp2
        Sopt = (1-lamb) * cal_affinity_score_for_all(Xopt, K) + lamb * torch.sqrt(torch.matmul(cp[:, k][:, None], cp[k, :][None, :]))+ beta*torch.sqrt(torch.matmul(cp2[:, k][:, None], cp2[k, :][None, :]))
        update = (Sopt > Sorg)[:, :, None, None]

        #update = update & diag
        update[np.diag_indices(num_graph)] = False
        X = update * Xopt + ~update * X

    X = X.detach().cpu().numpy()
    return X

    pass
