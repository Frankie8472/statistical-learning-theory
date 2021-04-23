import sklearn as skl
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize
from sklearn.utils.validation import check_is_fitted

import pandas as pd
import numpy as np
from treelib import Tree

import matplotlib.pyplot as plt


def read_data_csv(sheet, y_names=None):
    """Parse a column data store into X, y arrays

    Args:
        sheet (str): Path to csv data sheet.
        y_names (list of str): List of column names used as labels.

    Returns:
        X (np.ndarray): Array with feature values from columns that are not
        contained in y_names (n_samples, n_features)
        y (dict of np.ndarray): Dictionary with keys y_names, each key
        contains an array (n_samples, 1) with the label data from the
        corresponding column in sheet.
    """

    data = pd.read_csv(sheet)
    feature_columns = [c for c in data.columns if c not in y_names]
    X = data[feature_columns].values
    y = dict([(y_name, data[[y_name]].values) for y_name in y_names])

    return X, y


class DeterministicAnnealingClustering(skl.base.BaseEstimator,
                                       skl.base.TransformerMixin):
    """Template class for DAC

    Attributes:
        cluster_centers (np.ndarray): Cluster centroids y_i
            (n_clusters, n_features)
        cluster_probs (np.ndarray): Assignment probability vectors
            p(y_i | x) for each sample (n_samples, n_clusters)
        bifurcation_tree (treelib.Tree): Tree object that contains information
            about cluster evolution during annealing.

    Parameters:
        n_clusters (int): Maximum number of clusters returned by DAC.
        random_state (int): Random seed.
    """

    def __init__(self, n_clusters=8, random_state=42, metric="euclidian"):
        self.random_state = random_state
        np.random.seed(self.random_state)

        self.n_clusters = n_clusters
        self.metric = metric
        self.T_max = 130
        self.T_min = None
        self.alpha = 0.95

        self.cluster_centers = None
        self.cluster_probs = None

        self.n_eff_clusters = list()
        self.temperatures = list()
        self.distortions = list()
        self.bifurcation_tree = Tree()

        # Not necessary, depends on your implementation
        self.bifurcation_tree_cut_idx = None

        # Add more parameters, if necessary. You can also modify any other
        # attributes defined above

    def fit(self, samples):
        """Compute DAC for input vectors X

        Preferred implementation of DAC as described in reference [1].

        Args:
            X (np.ndarray): Input array with shape (samples, n_features)
        """
        # TODO:
        dim = len(samples[0])
        N = len(samples)
        self.cluster_centers = []

        pca = PCA(svd_solver='full')
        pca.fit(samples)
        T = 3 * pca.explained_variance_[0]  # Start Temperature, 3 > 2

        self.cluster_centers.append(samples.mean(axis=0))  # y
        self.temperatures.append(T)
        self.T_min = T * (self.alpha ** self.T_max)
        K = 1
        p_y = [1]

        while T != 0.0001:
            if T <= self.T_min:
                T = 0.0001

            # UPDATE
            distances = np.zeros((N, K))
            for x in range(N):
                for j in range(K):
                    distances[x, j] = np.exp(-np.sum((samples[x] - self.cluster_centers[j]) ** 2) / T)

            converged = False
            max_iter = 1000
            iter = 0

            while not converged and iter < max_iter:
                converged = True
                iter += 1

                for i in range(K):
                    p_y_ = p_y.copy()
                    p_yi_x = np.zeros(N)
                    for x in range(N):
                        p_yi_x[x] = p_y_[i] * distances[x, i] / np.inner(p_y_, distances[x])

                    p_y[i] = np.sum(p_yi_x) / N

                    new_y_i = p_yi_x @ samples / (N * p_y[i])

                    if np.sqrt(np.sum((self.cluster_centers[i] - new_y_i)**2)) > 0.003 and converged:
                        converged = False

                    self.cluster_centers[i] = new_y_i

            T *= self.alpha

            if K < self.n_clusters:
                distances = np.zeros((N, K))
                Z_x = np.zeros((N, K))
                new_clusters = 0
                for x in range(N):
                    for j in range(K):
                        distances[x, j] = np.exp(-np.sum((samples[x] - self.cluster_centers[j]) ** 2) / T)
                    Z_x[x] = np.inner(p_y, distances[x])
                for i in range(K):
                    C_pos = np.zeros((dim, dim))
                    for x in range(N):
                        C_pos += distances[x, i] / Z_x[x] * np.cov(samples[x] - self.cluster_centers[i])
                    C_pos = p_y[i] * C_pos / N

                    e_val = np.linalg.eigvalsh(C_pos)[-1]

                    if T <= 2 * e_val and K + new_clusters < self.n_clusters:
                        self.cluster_centers.append(self.cluster_centers[i] + np.random.normal(0, 0.001, size=dim))
                        p_y[i] /= 2
                        p_y.append(p_y[i])
                        new_clusters += 1

                K += new_clusters

        self.T_min = T
        return self

    def _calculate_cluster_probs(self, dist_mat, temperature):
        """Predict assignment probability vectors for each sample in X given
            the pairwise distances

        Args:
            dist_mat (np.ndarray): Distances (n_samples, n_centroids)
            temperature (float): Temperature at which probabilities are
                calculated

        Returns:
            probs (np.ndarray): Assignment probability vectors
                (new_samples, n_clusters)
        """
        # TODO:
        return normalize(np.exp(-np.sum(dist_mat, axis=0)**2/temperature), norm='l1')

    def get_distance(self, samples, clusters):
        """Calculate the distance matrix between samples and codevectors
        based on the given metric

        Args:
            samples (np.ndarray): Samples array (n_samples, n_features)
            clusters (np.ndarray): Codebook (n_centroids, n_features)

        Returns:
            D (np.ndarray): Distances (n_samples, n_centroids)
        """
        # TODO:
        return pairwise_distances(samples, clusters, metric=self.metric)

    def predict(self, samples):
        """Predict assignment probability vectors for each sample in X.

        Args:
            samples (np.ndarray): Input array with shape (new_samples, n_features)

        Returns:
            probs (np.ndarray): Assignment probability vectors
                (new_samples, n_clusters)
        """
        distance_mat = self.get_distance(samples, self.cluster_centers)
        probs = self._calculate_cluster_probs(distance_mat, self.T_min)
        return probs

    def transform(self, samples):
        """Transform X to a cluster-distance space.

        In the new space, each dimension is the distance to the cluster centers

        Args:
            samples (np.ndarray): Input array with shape
                (new_samples, n_features)

        Returns:
            Y (np.ndarray): Cluster-distance vectors (new_samples, n_clusters)
        """
        check_is_fitted(self, ["cluster_centers"])

        distance_mat = self.get_distance(samples, self.cluster_centers)
        return distance_mat

    def plot_bifurcation(self):
        """Show the evolution of cluster splitting

        This is a pseudo-code showing how you may be using the tree
        information to make a bifurcation plot. Your implementation may be
        entire different or based on this code.
        """
        check_is_fitted(self, ["bifurcation_tree"])

        clusters = [[] for _ in range(len(np.unique(self.n_eff_clusters)))]
        for node in self.bifurcation_tree.all_nodes_itr():
            c_id = node.data['cluster_id']
            my_dist = node.data['distance']

            if c_id > 0 and len(clusters[c_id]) == 0:
                clusters[c_id] = list(np.copy(clusters[c_id - 1]))
            clusters[c_id].append(my_dist)

        # Cut the last iterations, usually it takes too long
        cut_idx = self.bifurcation_tree_cut_idx + 20

        beta = [1 / t for t in self.temperatures]
        plt.figure(figsize=(10, 5))
        for c_id, s in enumerate(clusters):
            plt.plot(s[:cut_idx], beta[:cut_idx], '-k',
                     alpha=1, c='C%d' % int(c_id),
                     label='Cluster %d' % int(c_id))
        plt.legend()
        plt.xlabel("distance to parent")
        plt.ylabel(r'$1 / T$')
        plt.title('Bifurcation Plot')
        plt.show()

    def plot_phase_diagram(self):
        """Plot the phase diagram

        This is an example of how to make phase diagram plot. The exact
        implementation may vary entirely based on your self.fit()
        implementation. Feel free to make any modifications.
        """
        t_max = np.log(max(self.temperatures))
        d_min = np.log(min(self.distortions))
        y_axis = [np.log(i) - d_min for i in self.distortions]
        x_axis = [t_max - np.log(i) for i in self.temperatures]

        plt.figure(figsize=(12, 9))
        plt.plot(x_axis, y_axis)

        region = {}
        for i, c in list(enumerate(self.n_eff_clusters)):
            if c not in region:
                region[c] = {}
                region[c]['min'] = x_axis[i]
            region[c]['max'] = x_axis[i]
        for c in region:
            if c == 0:
                continue
            plt.text((region[c]['min'] + region[c]['max']) / 2, 0.2,
                     'K={}'.format(c), rotation=90)
            plt.axvspan(region[c]['min'], region[c]['max'], color='C' + str(c),
                        alpha=0.2)
        plt.title('Phases diagram (log)')
        plt.xlabel('Temperature')
        plt.ylabel('Distortion')
        plt.show()
