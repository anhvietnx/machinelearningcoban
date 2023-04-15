# %reset
import numpy as np 
from mnist.loader import MNIST
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from display_network import *


mndata = MNIST('/home/anhvietnx1/project/Project1/MNIST') # path to your MNIST folder 
mndata.load_testing()  #loads the testing images and labels from the MNIST dataset.
X = mndata.test_images
X0 = np.asarray(X)[:1000,:]/256.0 #lấy 1000 mẫu đầu tiên từ X , chia 256 để chuẩn hóa về 0 or 1
#kích thước 1000,784 do mỗi mẫu có kích thước 1,784
X = X0
K = 10
kmeans = KMeans(n_clusters=K, n_init=10).fit(X)

pred_label = kmeans.predict(X)

print(type(kmeans.cluster_centers_.T))
print(kmeans.cluster_centers_.T.shape)
A = display_network(kmeans.cluster_centers_.T, K, 1)

f1 = plt.imshow(A, interpolation='nearest', cmap = "jet")
f1.axes.get_xaxis().set_visible(False)
f1.axes.get_yaxis().set_visible(False)
plt.show()
# plt.savefig('a1.png', bbox_inches='tight')


# a colormap and a normalization instance
cmap = plt.cm.jet
norm = plt.Normalize(vmin=A.min(), vmax=A.max())

# map the normalized data to colors
# image is now RGBA (512x512x4) 
image = cmap(norm(A))

import imageio
image = cmap(norm(A))
image = (image * 255).astype(np.uint8) # convert to uint8
imageio.imwrite('aa.png', image)

#chon vai anh tu cluster
print(type(pred_label))
print(pred_label.shape)
print(type(X0))

N0 = 20;
X1 = np.zeros((N0*K, 784))
X2 = np.zeros((N0*K, 784))

for k in range(K):
    Xk = X0[pred_label == k, :]

    center_k = [kmeans.cluster_centers_[k]]
    neigh = NearestNeighbors(n_neighbors=N0).fit(Xk)

    dist, nearest_id  = neigh.kneighbors(center_k, N0)
    
    X1[N0*k: N0*k + N0,:] = Xk[nearest_id, :]
    X2[N0*k: N0*k + N0,:] = Xk[:N0, :]

plt.axis('off')
A = display_network(X2.T, K, N0)
f2 = plt.imshow(A, interpolation='nearest' )
plt.gray()
plt.show()

import imageio

# Save image
imageio.imwrite('bb.png', A)

# Display image
plt.axis('off')
A = display_network(X1.T, 10, N0)
imageio.imwrite('cc.png', A)
f2 = plt.imshow(A, interpolation='nearest' )
plt.gray()
plt.show()
