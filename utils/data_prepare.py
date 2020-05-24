import scipy.io as scio
import math
import random
import numpy as np
import os


def knlPQ2K(KP, KQ, Eg1, Eg2, n1, n2, m1, m2, nn):
    I11 = np.repeat(Eg1[0].reshape(-1, 1), m2, axis=1)
    I12 = np.repeat(Eg1[1].reshape(-1, 1), m2, axis=1)
    I21 = np.repeat(Eg2[0].reshape(1, -1), m1, axis=0)
    I22 = np.repeat(Eg2[1].reshape(1, -1), m1, axis=0)

    idx1 = np.array(
        [np.ravel_multi_index((x, y), dims=(n1, n2), order='F') for x, y in zip(I11.reshape(-1), I21.reshape(-1))])
    idx2 = np.array(
        [np.ravel_multi_index((x, y), dims=(n1, n2), order='F') for x, y in zip(I12.reshape(-1), I22.reshape(-1))])
    vals = KQ.reshape(-1)
    idx1 = np.concatenate([idx1, np.arange(nn)], axis=0)
    idx2 = np.concatenate([idx2, np.arange(nn)], axis=0)
    vals = np.concatenate([vals, KP.reshape(-1)], axis=0)
    K = np.zeros([nn, nn])
    for i in range(vals.shape[0]):
        K[idx1[i]][idx2[i]] = vals[i]
    return K


def conKnlGphKU(KP, KQ, Eg1, Eg2):
    n1, n2 = KP.shape
    m1, m2 = KQ.shape
    nn = n1 * n2

    return knlPQ2K(KP, KQ, Eg1, Eg2, n1, n2, m1, m2, nn)


def conDst(X1, X2, bAngle):
    n1 = X1.shape[1]
    n2 = X2.shape[1]
    X1 = np.concatenate([X1, np.zeros([1, n1])], axis=0)
    X2 = np.concatenate([X2, np.zeros([1, n2])], axis=0)

    XX1 = np.sum(X1 * X1, axis=0)
    XX2 = np.sum(X2 * X2, axis=0)
    X12 = np.transpose(X1).dot(X2)

    D = np.repeat(XX1.reshape(-1, 1), n2, axis=1) + np.repeat(XX2.reshape(1, -1), n1, axis=0) - 2 * X12
    if bAngle:
        idx = np.nonzero(D > 1)
        D[idx] = np.square(np.sqrt(D[idx]) - 2)

    return D


class Data:
    def __init__(self, num_nodes=10, point_data=None):
        self.num_nodes = num_nodes
        self.nP = self.num_nodes
        self.edge = np.zeros([self.num_nodes, self.num_nodes], dtype=float)
        self.point = point_data
        self.angle = np.zeros([self.nP, self.nP])
        self.get_edge_and_angle()
        self.edgeRaw = None
        self.angleRaw = None
        self.edgeFeat = None
        self.angleFeat = None

    def get_edge_and_angle(self):
        nodeCnt = self.nP
        for r in range(nodeCnt):
            for c in range(r + 1, nodeCnt):
                self.edge[r][c] = np.sqrt(np.square(self.point[r][0] - self.point[c][0]) +
                                          np.square(self.point[r][1] - self.point[c][1]))
                if np.square(self.point[r][0] - self.point[c][0]) == 0:
                    self.angle[r][c] = 90 if self.point[r][1] - self.point[c][1] else -90
                else:
                    self.angle[r][c] = 180. / np.pi * math.atan((self.point[r][1] - self.point[c][1]) /
                                                                (self.point[r][0] - self.point[c][0]))
        self.edge = self.edge / np.max(self.edge)
        self.edge += np.transpose(self.edge)

        self.angle /= 90.
        self.angle += np.transpose(self.angle)

        self.edgeRaw = self.edge
        self.angleRaw = self.angle
        from scipy.spatial import Delaunay
        tri = Delaunay(self.point[:self.num_nodes])
        self.adjMatrix = np.zeros([self.nP, self.nP])
        tri = tri.simplices
        triNum = tri.shape[0]
        for i in range(triNum):
            self.adjMatrix[tri[i][0]][tri[i][1]] = 1
            self.adjMatrix[tri[i][1]][tri[i][0]] = 1
            self.adjMatrix[tri[i][0]][tri[i][2]] = 1
            self.adjMatrix[tri[i][2]][tri[i][0]] = 1
            self.adjMatrix[tri[i][1]][tri[i][2]] = 1
            self.adjMatrix[tri[i][2]][tri[i][1]] = 1


class Affinity:
    def __init__(self):
        self.EG = None
        self.nP = None
        self.G = None
        self.H = None
        self.edge = None
        self.edgeRaw = None
        self.angleRaw = None
        self.adj = None


class DataGenerator:
    def __init__(
            self,
            num_inlier=10,
            num_outlier=2,
            num_graphs=8,
            scale_2D=0.1,
            edgeAffinityWeight=0.9,
            angleAffinityWeight=0.1,
            data_path=None
    ):
        self.num_inlier = num_inlier
        self.num_outlier = num_outlier
        self.num_nodes = self.num_inlier + self.num_outlier
        self.num_graphs = num_graphs
        self.data_path = data_path
        self.coord_data_list = self.read_mat_and_sample()
        self.scale_2D = scale_2D
        self.edgeAffinityWeight = edgeAffinityWeight
        self.angleAffinityWeight = angleAffinityWeight

        self.data = []
        self.adjlen = []
        self.affinity = []
        self.KP = np.zeros([self.num_graphs, self.num_graphs, self.num_nodes, self.num_nodes])
        self.K = np.zeros(
            [self.num_graphs, self.num_graphs, self.num_nodes * self.num_nodes, self.num_nodes * self.num_nodes])
        _ = self.preprocess()
        self.gt = np.diagflat(np.ones(self.num_nodes)).reshape(1, 1, self.num_nodes, self.num_nodes)
        self.gt = np.tile(self.gt, (self.num_graphs, self.num_graphs, 1, 1))

    def preprocess(self, coord_data_list=None):
        if coord_data_list is None:
            coord_data_list = self.coord_data_list
        for i, point_data in enumerate(coord_data_list):
            self.data.append(Data(self.num_nodes, np.transpose(point_data)))
            self.adjlen.append(np.sum(self.data[-1].adjMatrix))

        for i in range(self.num_graphs):
            self.data[i].adjMatrix = self.data[i].adjMatrix.astype(bool)
            self.data[i].nE = np.sum(self.data[i].adjMatrix)
            rc = np.argwhere(self.data[i].adjMatrix == 1)
            self.data[i].edgeFeat = self.data[i].edge[self.data[i].adjMatrix.nonzero()].reshape(1, -1)
            self.data[i].angleFeat = self.data[i].angle[self.data[i].adjMatrix.nonzero()].reshape(1, -1)

            tmp_aff = Affinity()
            tmp_aff.EG = np.transpose(rc)  # 2 * edge
            tmp_aff.nP = self.data[i].nP
            tmp_aff.edge = self.data[i].edge
            tmp_aff.edgeRaw = self.data[i].edgeRaw
            tmp_aff.angleRaw = self.data[i].angleRaw
            tmp_aff.adj = self.data[i].adjMatrix

            self.affinity.append(tmp_aff)

        for i in range(self.num_graphs):
            for j in range(self.num_graphs):
                if i == j:
                    n = self.num_nodes
                    for ii in range(n):
                        for jj in range(n):
                            self.K[i][j][ii * n + ii][jj * n + jj] = 0.25
                    continue

                dq = np.zeros([self.data[i].edgeFeat.shape[1], self.data[j].edgeFeat.shape[1]])

                dq += self.edgeAffinityWeight * conDst(self.data[i].edgeFeat, self.data[j].edgeFeat, 0)
                dq += self.angleAffinityWeight * conDst(self.data[i].angleFeat, self.data[j].angleFeat, 1)
                dq = np.exp(-dq / self.scale_2D)
                self.K[i][j] = conKnlGphKU(self.KP[i][j], dq, self.affinity[i].EG, self.affinity[j].EG)
        return self.K

    def make_assert(self, i, j, i1, j1, a1, b1, n):
        return self.K[i][j][j1 * n + i1][b1 * n + a1] == self.K[j][i][i1 * n + j1][a1 * n + b1] == \
               self.K[i][j][b1 * n + a1][j1 * n + i1] == self.K[i][j][b1 * n + i1][j1 * n + a1]

    def read_mat_and_sample(self):
        tmp = []
        data_list = [item for item in os.listdir(self.data_path)]
        data_list.sort()
        for i, item in enumerate(data_list):
            if item == "pairwise_matching.mat":
                mat_path = '{}/{}'.format(self.data_path, item)
                data = np.array(scio.loadmat(mat_path)['rawMat'], dtype=float)
                m = data.shape[0] // self.num_inlier
                n = self.num_inlier
                init_mat = np.zeros((m, m, n, n))
                for vi in range(m):
                    for vj in range(m):
                        init_mat[vi][vj] = data[vi * n:(vi + 1) * n, vj * n:(vj + 1) * n]
            elif item[-3:] == "mat":
                mat_path = '{}/{}'.format(self.data_path, item)
                data = np.array(scio.loadmat(mat_path)['pts_coord'], dtype=float)
                tmp.append(data)
        tmp = random.sample(tmp, self.num_graphs)

        for i, coord in enumerate(tmp):
            max_x, min_x = np.max(coord[0]), np.min(coord[0])
            max_y, min_y = np.max(coord[1]), np.min(coord[1])
            tmp_outlier = np.random.rand(2, self.num_outlier)
            tmp_outlier[0] = 2 * (tmp_outlier[0] - 0.5) * (max_x - min_x) + (min_x + max_x) / 2.
            tmp_outlier[1] = 2 * (tmp_outlier[1] - 0.5) * (max_y - min_y) + (min_y + max_y) / 2.
            tmp[i] = np.concatenate([tmp[i], tmp_outlier], axis=1)

        return tmp
