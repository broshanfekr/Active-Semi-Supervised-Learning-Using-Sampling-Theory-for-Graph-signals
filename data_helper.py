from scipy import spatial
import numpy as np
import scipy.io as sio
from numpy.random import permutation


def load_dataset(myargs):
    img, label = load_USPS_data_instace("USPSdata/usps_all.mat", myargs)
    img = np.reshape(img, [img.shape[0], -1])

    # building the adjacency matrix
    print("building adjacency matrix")
    adj_mat, sigma = build_knn_adj_mat(img, k=myargs.k)

    return img, label, adj_mat, sigma


def load_USPS_data_instace(path, myargs):
    data = sio.loadmat(path)
    img = data['data']
    img_data = []
    label = []
    for i in range(img.shape[-1]):
        temp_sample_list = []
        for j in range(img.shape[1]):
            temp = np.reshape(img[:, j, i], [16, 16])
            temp_sample_list.append(temp)

        idx = permutation(len(temp_sample_list))
        idx = idx[:myargs.sample_per_class]
        selected_samples = [temp_sample_list[x] for x in idx]
        selected_labels = [i for _ in idx]
        img_data.extend(selected_samples)
        label.extend(selected_labels)

    img_data = np.array(img_data)
    img_data = img_data.astype('float')
    label = np.array(label[:])
    label = to_onehot(label)
    return img_data, label


def build_knn_adj_mat(data_nodes, k):
    shape = data_nodes.shape
    # compute pairwise distances
    distance_matrix = np.zeros([shape[0], shape[0]])
    # Assign lower triangular part only in loop, saves time
    for i in range(shape[0]):
        for j in range(i):
            distance_matrix[i, j] = spatial.distance.euclidean(data_nodes[i, :],
                                                               data_nodes[j, :])
    # Complete upper triangular part
    distance_matrix = np.add(distance_matrix, distance_matrix.T)

    # Calculating distances of k-nearest neighbors
    # sort all possible neighbors according to distance
    knn_distance = np.sort(distance_matrix, axis=1)

    # sparsification matrix
    nodes_to_retain = np.ones([shape[0], shape[0]])

    for i in range(shape[0]):
        nodes_to_retain[i, distance_matrix[i, :] > knn_distance[i, k]] = 0
        nodes_to_retain[i, i] = 0  # diagonal should be zero

    nodes_to_retain[nodes_to_retain != nodes_to_retain.T] = 1

    # keep distances in the range of knn distance
    distance_matrix = np.multiply(distance_matrix, nodes_to_retain)

    # computing sigma
    sigma = 1/3 * np.mean(knn_distance[:, k])

    # build knn adjacency matrix
    adjacency_mat = np.zeros([shape[0], shape[0]])
    for i in range(shape[0]):
        for j in range(i):
            if distance_matrix[i, j] == 0:
                continue
            else:
                adjacency_mat[i, j] = np.exp((-1.0 * distance_matrix[i, j] ** 2) / (2 * sigma ** 2))
    adjacency_mat = adjacency_mat + adjacency_mat.T

    return adjacency_mat, sigma


def to_onehot(label):
    m = len(set(label))
    n = len(label)
    onehot_matrix = np.zeros([n, m])
    for i in range(n):
        onehot_matrix[i, label[i]] = 1
    return onehot_matrix


def build_normalized_laplacian_matrix(adjacency_matrix):
    # compute the symmetric normalized Laplacian matrix
    d = np.sum(adjacency_matrix, axis=1, dtype=float)
    for i in range(len(d)):
        if d[i] != 0:
            d[i] = d[i] ** (-1.0/2)
    Dinv = np.diag(d)
    Ln = np.eye(len(d)) - np.matmul(np.matmul(Dinv, adjacency_matrix), Dinv)
    # make sure the Laplacian is symmetric
    Ln = 0.5 * (Ln + Ln.T)
    return Ln