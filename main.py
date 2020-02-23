from k_means import init_centroids
import numpy as np
import math
import matplotlib.pyplot as plt
from imageio import imread


def calcCentroids(X, k):
    cor_x = list()
    cor_y = list()
    print('k=%d:' % k)
    centroids = init_centroids.init_centroids(X, k)
    # print centroida - iteration 0
    print('iter 0:', print_cent(centroids))

    for i in range(1, 11):
        clusters = {}
        # init the clusters
        for j in range(k):
            clusters[j] = list()

        # for each pixel-find the closest centroid
        for pixel in X:
            distances = list()
            # distance from each centroid
            for j, centroid in enumerate(centroids):
                distances.append(calc_distance(centroid, pixel))
            centroidIndex = distances.index(min(distances))
            clusters[centroidIndex].append(pixel)

        # cala loss function
        """loss = 0
        for j, centroid in enumerate(centroids):
            for m in range(len(clusters[j])):
                loss += calc_distance(centroid, clusters[j][m])

        loss /= len(X)
        cor_x.append(i-1)
        cor_y.append(loss)"""

        for centroidIndex in clusters:
            centroids[centroidIndex] = np.average(clusters[centroidIndex], axis=0)

        print('iter %d:' %i,print_cent(centroids))


    """plt.plot(cor_x,cor_y)
    plt.title("k = %d" %k)
    plt.xlabel('iter')
    plt.ylabel('loss')
    plt.show()
    create_pic(X, centroids)"""


def load_data():
    # data preperation (loading, normalizing, reshaping)
    path = 'dog.jpeg'
    A = imread(path)
    A_norm = A.astype(float) / 255.
    img_size = A_norm.shape
    X = A_norm.reshape(img_size[0] * img_size[1], img_size[2])
    return X


def plotPic(X):
    A_norm = X.reshape(128,128, 3)
    # plot the image
    plt.imshow(A_norm)
    plt.grid(False)
    plt.show()

# picture by centroids
def create_pic(X, centroids):
    for i, pixel in enumerate(X):
        distances = list()
        for j, centroid in enumerate(centroids):
            distances.append(calc_distance(centroid, pixel))
        index = distances.index(min(distances))
        X[i] = centroids[index]
    plotPic(X)


# calc euclidian distance
def calc_distance(centroid, pixel):
    distance = 0
    for i in range(len(centroid)):
        distance += pow((centroid[i] - pixel[i]), 2)
    distance = math.sqrt(distance)
    return pow(distance,2)


def print_cent(cent):
    if type(cent) == list:
        cent = np.asarray(cent)
    if len(cent.shape) == 1:
        return ' '.join(str(np.floor(100 * cent) / 100).split()).replace('[ ', '[').replace('\n', ' ').replace(' ]',
                                                                                                               ']').replace(
            ' ', ', ')
    else:
        return ' '.join(str(np.floor(100 * cent) / 100).split()).replace('[ ', '[').replace('\n', ' ').replace(' ]',
                                                                                                               ']').replace(
            ' ', ', ')[1:-1]


if __name__ == '__main__':
    for k in [2, 4, 8, 16]:
        calcCentroids(load_data(), k)
