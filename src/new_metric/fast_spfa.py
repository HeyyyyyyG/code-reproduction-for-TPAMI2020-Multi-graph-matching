def fast_spfa(K, X, num_graph, num_node):
    """
    :param K: affinity matrix, (num_graph, num_graph, num_node^2, num_node^2)
    :param X: matching results, X[:-1, :-1] is the matching results obtained by last iteration of MGM-SPFA,
              X[num_graph,:] and X[:,num_graph] is obtained via two-graph matching solver(RRWM), We suppose the last
              graph is the new coming graph. (num_graph, num_graph, num_node, num_node)
    :param num_graph: number of graph, int
    :param num_node: number of node, int
    :return: X, matching results, match graph_m to {graph_1, ... , graph_m-1)
    """
    import numpy as np
    import torch
    lamb_fast = 0.3
    C_min = 3
    beta = 0.2
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    X = torch.Tensor(X).to(device)
    K = torch.Tensor(K).to(device)

    def cal_affinity_for_one(X, K, i, j):
        x_vec = X[i][j].reshape(num_node*num_node, 1, order='F')
        aff = torch.matmul(torch.matmul(x_vec.T, K[i][j]), x_vec)
        return aff / X_norm

    def cal_pairwise_consistency2(X):
        num_graph = X.size(0)
        num_node = X.size(2)
        #cp2=torch.ones([num_graph,num_graph]).to(device)
        cp3=torch.ones([num_graph,num_graph]).to(device)

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
        vector_x = XT.reshape(X.size(0), X.size(1), -1, 1) #(num_graph, num_graph, num_node^2,1)
        vector_xT = vector_x.transpose(2,3)
        affinity_score = torch.matmul(torch.matmul(vector_xT, K), vector_x)  #(num_graph, num_graph, 1, 1)
        global X_norm
        X_norm = torch.max(affinity_score)
        normalized_affinity_score = affinity_score.reshape(X.size(0), X.size(1)) / X_norm #(num_graph, num_graph)
        return normalized_affinity_score

    Batch_num = max(1,num_graph//C_min)

    for k in range(num_graph):#affinity boost
        Xopt = torch.matmul(X[:, k, None], X[k, None, :])
        Sorg = cal_affinity_score_for_all(X, K)
        Sopt = cal_affinity_score_for_all(Xopt, K)
        update = (Sopt > Sorg)[:, :, None, None]

        update[np.diag_indices(num_graph)] = False

        X = update * Xopt + ~update * X

    for j in range(Batch_num):
        if j < Batch_num - 1:
            b = [i for i in range(j * C_min, (j + 1) * C_min)]
            if num_graph - 1 not in b:
                b.append(num_graph - 1)
        else:
            b = [i for i in range(j * C_min, num_graph)]
        Xc = X[b, ...][:, b]
        Kc = K[b, ...][:, b]
        #print(Xc.shape,Kc.shape)
        Q = [i for i in range(len(b) - 1)]
        # random.shuffle(Q)
        count = 0

        cpc = cal_pairwise_consistency(Xc)
        cpc2 = cal_pairwise_consistency2(Xc)
        mask = torch.zeros(Xc.size(0),Xc.size(1)).bool().to(device)# only update with N
        mask[:,-1]=True
        mask[-1,:]=True
        #print(mask)
        while(len(Q)>0):
            x = Q.pop(0)
            count+=1

            Xopt = torch.matmul(Xc[:, x, None], Xc[x, None, :])  # X_opt[y,N]=X[y,x]·X[x,N]
            Sorg = (1-lamb_fast) * cal_affinity_score_for_all(Xc, Kc) + lamb_fast * cpc + beta * cpc2
            Sopt = (1-lamb_fast) * cal_affinity_score_for_all(Xopt, Kc) + lamb_fast * torch.sqrt(torch.matmul(cpc[:, x][:, None], cpc[x, :][None, :])) + beta * torch.sqrt(torch.matmul(cpc2[:, x][:, None], cpc2[x, :][None, :]))
            update = (Sopt > Sorg)[:, :, None, None]
            update[np.diag_indices(Xc.size(0))] = False
            #print('update size',update.size())
            update = update & mask[:,:,None,None]
            #print(x,update)
            Xc = update * Xopt + ~update * Xc
            Q_new = (update.reshape(Xc.size(0),Xc.size(1))[:,-1]==1).nonzero()
            for i in Q_new:
                if i.item() not in Q:
                    Q.append(i.item())
            #print(Q)

            if count>Xc.size(0)*Xc.size(1):#num_graph*num_graph:
                break
        X[b, ...][:, b] = Xc
        '''
        while (len(Q) > 0):
            x = Q.pop(0)
            Q_new = []
            count += 1

            Xopt = np.matmul(Xc[:, x, None], Xc[x, None, :])

            for y in range(len(b) - 1):
                # print('y=',y)
                if y != x:
                    S_org = (1 - lamb_fast) * cal_affinity_for_one(Xc, Kc, y, -1) + lamb_fast * cpc[y][-1]
                    S_opt = (1 - lamb_fast) * cal_affinity_for_one(Xopt, Kc, y, -1) + lamb_fast * np.sqrt(
                        cpc[y][x] * cpc[x][-1])
                    if S_org < S_opt:
                        Xc[y, -1] = Xopt[y, -1]
                        if y not in Q:
                            Q_new.append(y)
            if count > C_min*C_min:
                break
        '''
    cp = cal_pairwise_consistency(X)
    cp2 = cal_pairwise_consistency2(X)
    for k in [num_graph-1]:
        Xopt = torch.matmul(X[:, k, None], X[k, None, :])
        Sorg = (1-lamb_fast) * cal_affinity_score_for_all(X, K) + lamb_fast * cp + beta * cp2
        Sopt = (1-lamb_fast) * cal_affinity_score_for_all(Xopt, K) + lamb_fast * torch.sqrt(torch.matmul(cp[:, k][:, None], cp[k, :][None, :])) + beta * torch.sqrt(torch.matmul(cp2[:, k][:, None], cp2[k, :][None, :]))
        update = (Sopt > Sorg)[:, :, None, None]
        update[np.diag_indices(num_graph)] = False
        X = update * Xopt + ~update * X

    X = X.detach().cpu().numpy()
    return X
    pass
