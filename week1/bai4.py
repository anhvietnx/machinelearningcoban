from __future__ import print_function 
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist # cdist được sử dụng để tính toán khoảng cách giữa các cặp điểm trong hai tập dữ liệu khác nhau
np.random.seed(11)

means = [[2, 2], [8, 3], [3, 6]] #các giá trị trung bình của các phân phối Gaussian đa chiều , các điểm sẽ phân phối xung quanh 
cov = [[1, 0], [0, 1]]   # Là ma trận hiệp phương sai (covariance matrix)
N = 500
X0 = np.random.multivariate_normal(means[0], cov, N) 
#print("x0=",X0)
X1 = np.random.multivariate_normal(means[1], cov, N)
#print(X1)
X2 = np.random.multivariate_normal(means[2], cov, N)
#print(X2)
# kết hợp ba tập dữ liệu X0, X1, và X2 thành một tập dữ liệu duy nhất X theo chieu doc
X = np.concatenate((X0, X1, X2), axis = 0) 

K = 3 # 3 clúter

original_label = np.asarray([0]*N + [1]*N + [2]*N).T
test=np.asarray([0]*10 + [1]*10 + [2]*10).T
#print(test)
#print(original_label)

def kmeans_display(X, label):
    K = np.amax(label) + 1 #trả về giá trị lớn nhất của mảng
    X0 = X[label == 0, :]
    X1 = X[label == 1, :]
    X2 = X[label == 2, :]
    
    plt.plot(X0[:, 0], X0[:, 1], 'b^', markersize = 4, alpha = .8)
    plt.plot(X1[:, 0], X1[:, 1], 'go', markersize = 4, alpha = .8)
    plt.plot(X2[:, 0], X2[:, 1], 'rs', markersize = 4, alpha = .8)

    plt.axis('equal')
    plt.plot()
    plt.show()
    
kmeans_display(X, original_label)

def kmeans_init_centers(X, k):
    # randomly pick k rows of X as initial centers
    return X[np.random.choice(X.shape[0], k, replace=False)]

def kmeans_assign_labels(X, centers):
    # calculate pairwise distances btw data and centers
    D = cdist(X, centers)
    # return index of the closest center
    return np.argmin(D, axis = 1)

def kmeans_update_centers(X, labels, K):
    centers = np.zeros((K, X.shape[1])) # khởi tạo ma trận centers với số hàng là K và số cột bằng số cột của X
    for k in range(K):
        # collect all points assigned to the k-th cluster 
        Xk = X[labels == k, :]
        # take average
        centers[k,:] = np.mean(Xk, axis = 0)
    return centers

def has_converged(centers, new_centers):
    # return True if two sets of centers are the same
    return (set([tuple(a) for a in centers]) == 
        set([tuple(a) for a in new_centers]))
    
def kmeans(X, K):
    centers = [kmeans_init_centers(X, K)]
    labels = []
    it = 0 
    while True:
        labels.append(kmeans_assign_labels(X, centers[-1]))
        new_centers = kmeans_update_centers(X, labels[-1], K)
        if has_converged(centers[-1], new_centers):
            break
        centers.append(new_centers)
        it += 1
    return (centers, labels, it)

(centers, labels, it) = kmeans(X, K)
print('Centers found by our algorithm:')
print(centers[-1])

kmeans_display(X, labels[-1])

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
print('Centers found by scikit-learn:')
print(kmeans.cluster_centers_)
pred_label = kmeans.predict(X) #nếu pred_label[i] bằng k, nghĩa là điểm X[i] được gán cho nhóm có nhãn k.
kmeans_display(X, pred_label)
