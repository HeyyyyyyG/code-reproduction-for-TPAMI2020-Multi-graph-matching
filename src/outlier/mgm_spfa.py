import numpy as np
import torch


def mgm_spfa(K, X, num_graph, num_node):
    """
    :param K: affinity matrix, (num_graph, num_graph, num_node^2, num_node^2)
    :param X: matching results, X[:-1, :-1] is the matching results obtained by last iteration of MGM-SPFA,
              X[num_graph,:] and X[:,num_graph] is obtained via two-graph matching solver(RRWM), We suppose the last
              graph is the new coming graph. (num_graph, num_graph, num_node, num_node)
    :param num_graph: number of graph, int
    :param num_node: number of node, int
    :return: X, matching results, match graph_m to {graph_1, ... , graph_m-1)
    """

    lamb_spfa = 0.25

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    X = torch.Tensor(X).to(device)
    K = torch.Tensor(K).to(device)

    outliers_indice=[]
    for g1 in [K.size(0)-1]:
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

    def cal_affinity_for_one(X,K,i,j):
        x_vec = X[i][j].reshape(num_node*num_node,1,order='F')
        aff = np.matmul(np.matmul(x_vec.T , K[i][j]) , x_vec)
        return aff/X_norm

    def cal_pairwise_consistency(X):
        """
        :param X: matching results, (num_graph, num_graph, num_node, num_node) torch tensor
        :return: pairwise_consistency: (num_graph, num_graph) torch tensor
        """
        for g1 in [X.size(0)-1]:
            X[g1,:,outliers_indice[g1],:]=0
        num_graph = X.size(0)
        num_node = X.size(2)
        X_t = X.transpose(0,1)
        pairwise_consistency = 1 - torch.abs(X[:, :, None] - torch.matmul(X[:, None], X_t[None, ...])).sum((2, 3, 4)) / (2 * num_graph * num_node)
        return pairwise_consistency

    def cal_affinity_score_for_all(X, K):
        #calculate affinity score for all pair of graphs
        # return: normalized_affinity_score: (num_graph, num_graph) torch tensor
        for g1 in [X.size(0)-1]:
            X[g1,:,outliers_indice[g1],:]=0
        XT = X.transpose(2,3)# in order for column-wise
        vector_x = XT.reshape(num_graph, num_graph, -1, 1) #(num_graph, num_graph, num_node^2,1)
        vector_xT = vector_x.transpose(2,3)
        affinity_score = torch.matmul(torch.matmul(vector_xT, K), vector_x)  #(num_graph, num_graph, 1, 1)
        global X_norm
        X_norm = torch.max(affinity_score)
        normalized_affinity_score = affinity_score.reshape(num_graph, num_graph) / X_norm #(num_graph, num_graph)
        return normalized_affinity_score

    for k in range(num_graph):
        Xopt = torch.matmul(X[:, k, None], X[k, None, :])
        Sorg = cal_affinity_score_for_all(X, K)
        Sopt = cal_affinity_score_for_all(Xopt, K)
        update = (Sopt > Sorg)[:, :, None, None]

        update[np.diag_indices(num_graph)] = False

        X = update * Xopt + ~update * X


    Q = [i for i in range(num_graph-1)]
    #random.shuffle(Q)
    count = 0

    cp = cal_pairwise_consistency(X)
    mask = torch.zeros(num_graph,num_graph).bool().to(device)# only update with N
    mask[:,-1]=True
    mask[-1,:]=True
    #print(mask)
    while(len(Q)>0):
        x = Q.pop(0)
        count+=1

        Xopt = torch.matmul(X[:, x, None], X[x, None, :])  # X_opt[y,N]=X[y,x]·X[x,N]
        Sorg = (1-lamb_spfa) * cal_affinity_score_for_all(X, K) + lamb_spfa * cp
        Sopt = (1-lamb_spfa) * cal_affinity_score_for_all(Xopt, K) + lamb_spfa * torch.sqrt(torch.matmul(cp[:, x][:, None], cp[x, :][None, :]))
        update = (Sopt > Sorg)[:, :, None, None]
        update[np.diag_indices(num_graph)] = False
        #print('update size',update.size())
        update = update & mask[:,:,None,None]
        #print(x,update)
        X = update * Xopt + ~update * X
        Q_new = (update.reshape(num_graph,num_graph)[:,-1]==1).nonzero()
        for i in Q_new:
            if i.item() not in Q:
                Q.append(i.item())
        #print(Q)

        if count>num_graph*num_graph:
            break
    #print(c)

    cp = cal_pairwise_consistency(X)
    #print('Q empty----------------------------------------------------------')

    for k in [num_graph-1]:
        Xopt = torch.matmul(X[:, k, None], X[k, None, :])
        Sorg = (1-lamb_spfa) * cal_affinity_score_for_all(X, K) + lamb_spfa * cp
        Sopt = (1-lamb_spfa) * cal_affinity_score_for_all(Xopt, K) + lamb_spfa * torch.sqrt(torch.matmul(cp[:, k][:, None], cp[k, :][None, :]))
        update = (Sopt > Sorg)[:, :, None, None]
        update[np.diag_indices(num_graph)] = False
        X = update * Xopt + ~update * X


    X = X.detach().cpu().numpy()
    return X

    pass

