from configuration import ArgValus
from scipy import special
import numpy as np
from scipy.sparse.linalg import svds, eigs

import chebyshev
import data_helper


def reconstruct_signal(Ln, observed_labels, sample_set, omega, myargs):
    filterlen = 10
    alpha = 8
    freq_range = [0, 2]

    chebyshev_coef = chebyshev.compute_cheby_coeff(lambda x: special.expit(alpha*(omega - x)), 
                                                    filterlen, filterlen+1, freq_range)
    
    f_hat = chebyshev.cheby_op(Ln, chebyshev_coef, observed_labels, freq_range)

    to_estimate_idx = sorted(list(set(range(Ln.shape[0])) - set(sample_set)))

    prev_diffrence = 0
    for i in range(myargs.num_iter):
        # projection on C1
        err_s = observed_labels - f_hat
        # error on the known set
        err_s[to_estimate_idx, :] = 0

        # projection on C2
        f_hat_temp = chebyshev.cheby_op(Ln, chebyshev_coef, f_hat + err_s, freq_range) # err on S approx LP

        # to check convergence
        diffrence = np.linalg.norm(f_hat_temp - f_hat)
        if i > 1 and diffrence > prev_diffrence:
            break
        else:
            prev_diffrence = diffrence
            f_hat = f_hat_temp

    return f_hat


def greedy_selection_of_samples(gl, num_of_samples):
    #  gl is the graph Laplacian matrix
    sample_set = []
    omega_list = []
    kth_power = 8

    Sc = set([i for i in range(gl.shape[0])])

    S = np.zeros([gl.shape[0], num_of_samples])

    # kth power of Laplacian
    print("computing proxies")
    L_K = np.linalg.matrix_power(gl, kth_power)
    L_K = 0.5 * (L_K + L_K.T)

    for i in range(num_of_samples):
        print("number of selected samples: ", len(sample_set))
        rc = sorted(list(Sc))
        reduced_L = L_K[rc, :]
        reduced_L = reduced_L[:, rc]

        eval, evec = eigs(reduced_L, k=1, sigma=10**(-20))

        omega = abs(eval) ** (1.0/kth_power)
        omega_list.append(omega)

        phi = np.absolute(evec)
        max_val = max(phi)
        max_index = np.argmax(phi)
        selected = rc[max_index]
        Sc = Sc - {selected}
        sample_set.append(selected)

    rc = sorted(list(Sc))
    reduced_L = L_K[rc, :]
    reduced_L = reduced_L[:, rc]
    eval, evec = eigs(reduced_L, k=1, sigma=10**(-20))
    omega = abs(eval) ** (1.0/kth_power)
    omega_list.append(omega)

    sample_set = sorted(sample_set)

    for i, sidx in enumerate(sample_set):
        S[sidx, i] = 1

    return S, sample_set, omega


def acc_score(pred_label, true_label, sample_set):
    Ind = np.argsort(-np.abs(pred_label), axis=1)
    pred_label = Ind[:, 0]

    Ind = np.argsort(-np.abs(true_label), axis=1)
    true_label = Ind[:, 0]

    all_samples = set(range(true_label.shape[0]))
    test_idx = list(all_samples - set(sample_set))

    test_acc = pred_label[test_idx] == true_label[test_idx]
    test_acc = sum(test_acc) / (len(true_label[test_idx]) * 1.0)

    train_acc = pred_label[sample_set] == true_label[sample_set]
    train_acc = sum(train_acc) / (len(true_label[sample_set]) * 1.0)

    return train_acc, test_acc


if __name__ == '__main__':
    myargs = ArgValus()

    img, label, adj_mat_p, sigma = data_helper.load_dataset(myargs)

    # compute the symmetric normalized Laplacian matrix
    Ln = data_helper.build_normalized_laplacian_matrix(adj_mat_p)

    # Choosing optimal sampling sets of different sizes
    print("sample selection from Graph...")
    S, sample_set, omega = greedy_selection_of_samples(Ln, num_of_samples=round(myargs.train_percent*Ln.shape[0]))

    # signal reconstruction
    observed_labels = np.zeros(label.shape)
    for sample in sample_set:
        observed_labels[sample, :] = label[sample, :]
    f_hat = reconstruct_signal(Ln, observed_labels, sample_set, omega, myargs)
    
    train_acc, test_acc = acc_score(f_hat, label, sample_set)
    print("train_acc is: {}    |   test_acc is: {}".format(train_acc, test_acc))

    print("the end")
