import numpy as np
import matplotlib.pyplot as plt

def d(u, v):
    diff = u -v
    return diff.dot(diff)

def cost(X, R, M):
    cost = 0
    for k in xrange(len(M)):
        for n in xrange(len(X)):
            cost += R[n, k]*d(M[k], X[n])
    return cost

def plot_k_means(X, K, max_iter=20, beta=1.0):
    N, D = X.shape
    M = np.zeros((K, D))
    R = np.ones((N, K)) / K

    # Initialize M to random

    for k in xrange(K):
        M[k] = X[np.random.choice(N)]

        grid_width = 5
        grid_height = max_iter / grid_width
        random_colors = np.random.random((K, 3))

        plt.figure()

        costs = np.zeros(max_iter)
        for i in xrange(max_iter):
            # moved plot inside for this loop
            colors = R.dot(random_colors)
        plt.subplot(grid_width, grid_height, i+1)
        plt.scatter(X[:, 0], X[:1], c=colors)

            # Step 1: determine assignments (possibly inefficient?)

        for k in xrange (K):
            for n in xrange (N):
                R[n,k] = np.exp(-beta*d(M[k], X[n])) / np.sum(np.exp(-beta*d(M[j], X[n])))
            for j in xrange(K):

            # Step 2: Recalculate means

                for k in xrange(K):
                    M[k] = R[:, k].dot(X) / R[:,k].sum()

            costs[i] = cost (X, R, M)
            if i > 0:
                if np.abs(costs[i] - costs[i-1]) < 10e-5: break

            plt.show()

        def main():
            # Assume 3 means
            D = 2 # To visualize easily
            s = 4 # Separation so we can control how far means are apart
            mu1 = np.array([0,0])
            mu2 = np.array([s, s])
            mu3 = np.array([0, s])

            N = 900 # Number of Samples
            X = np.zeros((N, D))
            X[:300, :] = np.random.randn(300, D) + mu1
            X[300:600, :] = np.random.randn(300, D) + mu2
            X[600:, :] = np.random.randn(300, D) + mu3

            # Without clustering

            plt.scatter(X[:,0],X[:,1])
            plt.show()

            K = 3 # We know this
            plot_k_means(X, K)

            # K = 5 # What happens if a "bad K is chosen
            # plot_k_means(X, K, mas_iter = 30)
            # K = 5 # Changing beta?
            # plot_k_means (X, K, max_iter = 30, beta = 0.3)

            if __name__ =='__main__':
                main()


