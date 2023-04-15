import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

img = mpimg.imread('girl3.jpg')
plt.imshow(img)
imgplot = plt.imshow(img)
plt.axis('off')
plt.show() 

#chuyển đổi kích thước của ảnh img từ (m, n, c) thành một ma trận 2 chiều X có kích thước (m * n, c). 
# Trong đó, m và n là chiều rộng và chiều cao của ảnh và c là số kênh màu 
X = img.reshape((img.shape[0]*img.shape[1], img.shape[2]))

for K in [2, 5, 10, 15, 20,100]:
    kmeans = KMeans(n_clusters=K, n_init='auto').fit(X)
    label = kmeans.predict(X)
    
    # array img4 which has the same shape as the input array X but all 
    # its elements are initialized to zero
    img4 = np.zeros_like(X)
    
    # replace each pixel by its center
    for k in range(K):
        img4[label == k] = kmeans.cluster_centers_[k]
    # reshape and display output image
    img5 = img4.reshape((img.shape[0], img.shape[1], img.shape[2]))
    plt.imshow(img5, interpolation='nearest')
    plt.axis('off')
    plt.show()
