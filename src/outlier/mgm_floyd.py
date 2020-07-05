import torch
import numpy as np
device = 'cpu'#''cuda' if torch.cuda.is_available() else 'cpu'

def mgm_floyd(X, K, num_graph, num_node):
    """
    :param K: affinity matrix, (num_graph, num_graph, num_node^2, num_node^2)
    :param num_graph: number of graph, int
    :param num_node: number of node, int
    :return: matching results, (num_graph, num_graph, num_node, num_node)
    """
    print(X.shape,K.shape,num_graph,num_node)

    lamb = 0.3
    #for i in K[0][1]:
     #   print(i.sum())

    X = torch.Tensor(X).to(device)
    K = torch.Tensor(K).to(device)
    outliers_indice = []
    '''#对角线极差
    for g1 in range(K.size(0)):
        ran = torch.zeros(num_node).to(device)
        for g2 in range(K.size(1)):
            diag_in_g1g2 = K[g1][g2].diag().reshape(num_node,num_node)
            print(diag_in_g1g2.sum(dim=1))
            min = diag_in_g1g2.sort()[0][:,0]
            max = diag_in_g1g2.sort()[0][:,-1]
            ran += -(max-min)
        outliers_indice.append(ran.topk(k=5)[1])
    print(outliers_indice)
    '''
    for g1 in range(K.size(0)):
        avg = torch.zeros(num_node).to(device)
        min = torch.zeros(num_node).to(device)
        max = torch.zeros(num_node).to(device)
        for g2 in range(K.size(1)):
            for i in range(num_node):
                avg[i] -= K[g1][g2][i*num_node:(i+1)*num_node].sum()
                min[i] -= (-K[g1][g2][i*num_node:(i+1)*num_node]).reshape(-1).topk(k=3)[0].sum()
                max[i] += K[g1][g2][i*num_node:(i+1)*num_node].reshape(-1).topk(k=3)[0].sum()
                #print(K[g1][g2][i*num_node:(i+1)*num_node,i*num_node:(i+1)*num_node].size(),K[g1][g2][i*num_node:(i+1)*num_node,i*num_node:(i+1)*num_node])
        #print(g1,ran)
        lamb_metric = 0.75
        avg = avg/avg.max()
        ran = max-min
        ran = ran/ran.max()
        metric = -((1-lamb_metric)*avg + lamb_metric * ran)
        outliers_indice.append(metric.topk(k=3)[1].long())
        #print('metric',metric.topk(k=3),'ran',ran.topk(k=3),'avg',avg.topk(k=3))
        #print(g1,outliers_indice)

    def cal_pairwise_consistency(X):
        """
        :param X: matching results, (num_graph, num_graph, num_node, num_node) torch tensor
        :return: pairwise_consistency: (num_graph, num_graph) torch tensor
        """
        for g1 in range(X.size(0)):
            X[g1,:,outliers_indice[g1],:]=0
        num_graph = X.size(0)
        num_node = X.size(2)
        X_t = X.transpose(0,1)
        pairwise_consistency = 1 - torch.abs(X[:, :, None] - torch.matmul(X[:, None], X_t[None, ...])).sum((2, 3, 4)) / (2 * num_graph * num_node)
        return pairwise_consistency


    def cal_affinity_score_for_all(X, K):
        #calculate affinity score for all pair of graphs
        # return: normalized_affinity_score: (num_graph, num_graph) torch tensor
        for g1 in range(X.size(0)):
            X[g1,:,outliers_indice[g1],:]=0
        #print(X.size())
        XT = X.transpose(2,3)# in order for column-wise
        vector_x = XT.reshape(num_graph, num_graph, -1, 1) #(num_graph, num_graph, num_node^2,1)
        vector_xT = vector_x.transpose(2,3)
        #print(vector_x.size(),vector_xT.size())
        affinity_score = torch.matmul(torch.matmul(vector_xT, K), vector_x)  #(num_graph, num_graph, 1, 1)
        #print(affinity_score)
        global X_norm
        X_norm = torch.max(affinity_score)
        normalized_affinity_score = affinity_score.reshape(num_graph, num_graph) / X_norm #(num_graph, num_graph)
        return normalized_affinity_score

    #diag = torch.eye(num_graph)<0.5
    #diag = diag.to(device)
    for k in range(num_graph):
        Xopt = torch.matmul(X[:, k, None], X[k, None, :])
        #print('XOPT',Xopt.size(),'X',X.size())
        Sorg = cal_affinity_score_for_all(X, K)
        Sopt = cal_affinity_score_for_all(Xopt, K)
        update = (Sopt > Sorg)[:, :, None, None]

        update[np.diag_indices(num_graph)] = False

        X = update * Xopt + ~update * X

    for k in range(num_graph):
        cp = cal_pairwise_consistency(X)
        Xopt = torch.matmul(X[:, k, None], X[k, None, :])
        Sorg = (1-lamb) * cal_affinity_score_for_all(X, K) + lamb * cp
        Sopt = (1-lamb) * cal_affinity_score_for_all(Xopt, K) + lamb * torch.sqrt(torch.matmul(cp[:, k][:, None], cp[k, :][None, :]))
        update = (Sopt > Sorg)[:, :, None, None]

        #update = update & diag
        update[np.diag_indices(num_graph)] = False
        X = update * Xopt + ~update * X

    X = X.detach().cpu().numpy()
    return X

    pass
