import numpy as np
import numba
from numba import njit



def mgm_floyd(X, K, num_graph, num_node):
    """
    :param K: affinity matrix, (num_graph, num_graph, num_node^2, num_node^2)
    :param num_graph: number of graph, int
    :param num_node: number of node, int
    :return: matching results, (num_graph, num_graph, num_node, num_node)
    """
    #print(X.shape,num_graph,num_node)

    lamb = 0.9


    def distance(X_g1g2,K_g1g2,cp_g1g2):
        x_vec = X_g1g2.reshape(num_node*num_node,1,order='F')
        aff = np.matmul(np.matmul(x_vec.T , K_g1g2) , x_vec)
        #print('aff',aff)
        #print(aff,cp_g1g2)
        return (1-lamb) * aff/20 + lamb * cp_g1g2



    def calculate_Cp(X):
        cp=np.ones([num_graph,num_graph])
        for i in range(num_graph):
            for j in range(num_graph):
                for k in range(num_graph):
                    cp[i][j] -= np.linalg.norm(X[i][j]-np.matmul(X[i][k],X[k][j]), ord='fro')/2/num_graph/num_graph
                #cp[i][j]=1-cp[i][j]

        return cp


    for g3 in range(num_graph):
        if g3%2==0:
            cp=calculate_Cp(X)
        for g1 in range(num_graph):
            for g2 in range(num_graph):
                #d_g1g2 = distance(X[g1][g2],K[g1][g2],cp[g1][g2])
                Xs_g1g2 = np.matmul(X[g1][g3],X[g3][g2])
                #Xs[g1][g2] = np.matmul(Xs[g1][g3],Xs[g3][g2])
                #print('change??????????',(X[g1][g2]==Xs_g1g2).all())
                #print('original',distance(X,K,g1,g2,cp[g1][g2]),'after',distance(Xs,K,g1,g2,cp[g1][g2]))

                x_vec1 = X[g1][g2].reshape(num_node*num_node,1,order='F')
                aff1 = np.matmul(np.matmul(x_vec1.T , K[g1][g2]) , x_vec1)
                distance1 = (1-lamb) * aff1/20 + lamb * cp[g1][g2]

                x_vec2 = Xs_g1g2.reshape(num_node*num_node,1,order='F')
                aff2 = np.matmul(np.matmul(x_vec2.T , K[g1][g2]) , x_vec2)
                distance2 = (1-lamb) * aff2/20 + lamb * np.power(cp[g1][g2]*cp[g1][g3],0.5)


                if distance2>distance1:#distance(Xs_g1g2,K[g1][g2],cp[g1][g2]) > distance(X[g1][g2],K[g1][g2],cp[g1][g2]):#
                    X[g1][g2]=Xs_g1g2
                    #print('change')

    return X

    pass
