import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, estimate_bandwidth
from sklearn.cluster import MeanShift
from sklearn.metrics import silhouette_score

data = np.loadtxt('lab1.txt', delimiter=',')

# Метод зсуву середнього для визначення кількості кластерів
def meanshift(data):
    def meanshift(data):
        bandwidth_data = estimate_bandwidth(data, quantile=0.14, n_samples=len(data))

        meanshift_model = MeanShift(bandwidth=bandwidth_data, bin_seeding=True)
        meanshift_model.fit(data)

        cluster_centers = meanshift_model.cluster_centers_
        print('\nCenters of clusters:\n', cluster_centers)

        # calculating clusters
        labels = meanshift_model.labels_
        num_clusters = len(np.unique(labels))
        print("\nNumbers of clusters in input data =", num_clusters)

        plt.figure()
        markers = 'o*xvs+p'
        for i, marker in zip(range(num_clusters), markers):
            plt.scatter(data[labels == i, 0], data[labels == i, 1], marker=marker, color='black')

            cluster_center = cluster_centers[i]
            plt.plot(cluster_center[0], cluster_center[1], marker=marker, markerfacecolor='black',
                     markeredgecolor='black', markersize=15)

        plt.title('Centers of clusters')
        plt.show()


# Оцінка кластеризації за допомогою silhouette_score
def evaluate_clusters(data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters)
    labels = kmeans.fit_predict(data)
    score = silhouette_score(data, labels)
    return score

# Знаходження оптимальної кількості кластерів
def find_optimal_clusters(data):
    scores = []
    for k in range(2, 16):
        score = evaluate_clusters(data, k)
        scores.append(score)

    # Рисуємо бар-діаграму для оцінки кластерів
    plt.figure(figsize=(8, 6))
    plt.bar(range(2, 16), scores, align='center')
    plt.title('Silhouette Score for Different Numbers of Clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.show()

    optimal_clusters = np.argmax(scores) + 2  # +2, тому що ми починаємо з k=2
    return optimal_clusters

# Відображення результатів кластеризації
def plot_clusters(data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters)
    labels = kmeans.fit_predict(data)

    # Рисуємо вихідні точки
    plt.figure(figsize=(8, 6))
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis')
    plt.title('Original Data')
    plt.show()

    # Рисуємо центри кластерів
    plt.figure(figsize=(8, 6))
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='X', s=200, c='red')
    plt.title('Cluster Centers')
    plt.show()

    # Рисуємо області кластеризації
    plt.figure(figsize=(8, 6))
    h = 0.02
    x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
    y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap='viridis', alpha=0.8)
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis')
    plt.title('Clustered Data with Decision Boundaries')
    plt.show()

# Застосуємо метод зсуву середнього для визначення кількості кластерів
meanshift(data)

# Знайдемо оптимальну кількість кластерів за допомогою silhouette_score
optimal_clusters = find_optimal_clusters(data)

# Відобразимо результати кластеризації
plot_clusters(data, optimal_clusters)
