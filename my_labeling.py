__authors__ = 'TO_BE_FILLED'
__group__ = 'TO_BE_FILLED'

import numpy as np
import numpy.lib.arraysetops as aso
import Kmeans as km
import KNN as kn
import pandas as pd
from skimage.transform import resize
from utils_data import read_dataset, visualize_k_means, visualize_retrieval
import matplotlib.pyplot as plt
import cv2
import time


def get_color_predictions(images, max_k):
    # preds = np.empty((len(images), k), dtype='<U8')
    preds = []

    for ix, input in enumerate(images):
        # S'ha observat que el nombre d'iteracions necessàries era proper a k*5. 
        # Si el sobrepassa, és que no està essent eficient
        # La tolerància podria ser 0.05 però no val la pena
        kms = km.KMeans(input, 2, {"km_init": "kmeans++", "max_iter": max_k*5, "threshold": 0.1, "fitting": "WCD"})
        kms.find_bestK(max_k)
        kms.fit()
        preds.append(km.get_colors(kms.centroids))

    return preds


def get_shape_predictions(train_images, test_images, train_labels, k):
    knn = kn.KNN(train_images, train_labels)
    preds = knn.predict(test_images, k)
    return preds


def retrieval_by_color(images, labels, keywords):
    # args = probs[:,0].argsort()
    # labels = labels[args][::-1]
    idx = []

    for i in range(labels.shape[0]):
        if set(keywords).issubset(labels[i]) or set(labels[i]).issubset(keywords):
            idx.append(i)

    return images[idx]


def retrieval_by_shape(images, labels, keyword):
    mask = np.where(keyword == labels)
    return images[mask]


def retrieval_combined(images, color_labels, shape_labels, color_keyword, shape_keyword):
    # args = color_probs[:,0].argsort()
    # color_labels = color_labels[args][::-1]
    # shape_labels = shape_labels[args][::-1]
    mask_color = []

    for i in range(len(color_labels)):
        if set(color_keyword).issubset(color_labels[i]) or set(color_labels[i]).issubset(color_keyword):
            mask_color.append(i)

    mask_shape = np.where(shape_keyword == shape_labels)
    idx = np.intersect1d(np.array(mask_color), mask_shape)
    
    return images[idx]


def kmean_statistics(images, kmax=10, nsamples=20):
    global_times = np.zeros((kmax-1))
    global_scores = np.zeros((kmax-1))
    global_iterations = np.zeros((kmax-1))

    for ix, input in enumerate(images[:nsamples]):
        kms = km.KMeans(input, 2, {"km_init": "kmeans++", "threshold": 0.8, "fitting": "WCD"})
        local_times = []
        local_scores = []
        local_iterations = []

        for k in range(2, kmax+1):
            start = time.time()
            kms.K = k
            kms.fit()
            end = time.time()
            elapsed = end - start
            score = kms.fisher_score()
            global_times[k-2] += elapsed
            global_scores[k-2] += score
            global_iterations[k-2] += kms.num_iter

            local_scores.append(score)
            local_iterations.append(kms.num_iter)
            local_times.append(elapsed)
            print("Results for image " + str(ix) + " with k=" + str(k)) 
            print("Score: " + str(score))
            print("Iterations needed: " + str(kms.num_iter))
            print("Elapsed time: " + str(elapsed))
            print("")
            # visualize_k_means(kms, input.shape)
        
        # score_series = pd.Series(local_scores, index=list(range(2,kmax+1)), name="Score")
        # score_series.plot(legend=True)
        # plt.show()
        # iterations_series = pd.Series(local_iterations, index=list(range(2,kmax+1)), name="Iterations")
        # iterations_series.plot(legend=True)
        # plt.show()
        # time_series = pd.Series(local_times, index=list(range(2,kmax+1)), name="Time")
        # time_series.plot(legend=True)
        # plt.show()

    global_scores /= images.shape[0]
    global_iterations /= images.shape[0]
    global_times /= images.shape[0]

    score_series = pd.Series(global_scores, index=list(range(2,kmax+1)), name="Score")
    score_series.plot(legend=True)
    plt.show()
    iterations_series = pd.Series(global_iterations, index=list(range(2,kmax+1)), name="Iterations")
    iterations_series.plot(legend=True)
    plt.show()
    time_series = pd.Series(global_times, index=list(range(2,kmax+1)), name="Time")
    time_series.plot(legend=True)
    plt.show()


def knn_statistics(train_images, test_images, train_labels, test_labels, kmax):
    kmax += 1
    acc_list = []
    times_list = []

    for k in range(2, kmax):
        start = time.time()
        preds = get_shape_predictions(train_images, test_images, train_labels, k)
        end = time.time()
        times_list.append(end-start)
        acc_list.append(get_shape_accuracy(preds, test_labels))

    acc_series = pd.Series(acc_list, index=list(range(2,kmax)), name="Accuracy")
    acc_series.plot(legend=True)
    time_series = pd.Series(times_list, index=list(range(2,kmax)), name="Time")
    time_series.plot(legend=True)
    plt.show()


def get_shape_accuracy(predicted, ground_truth):
    return np.sum(predicted == ground_truth).astype("float") / predicted.shape[0] * 100


def get_color_accuracy(predicted, ground_truth):
    sumup = 0.0

    for i in range(len(predicted)):
        if set(predicted[i].tolist()).issubset(ground_truth[i]) or set(ground_truth[i]).issubset(predicted[i].tolist()):
            sumup += 1

    return sumup / len(predicted) * 100


if __name__ == '__main__':
    #Load all the images and GT
    train_imgs, train_class_labels, train_color_labels, \
    test_imgs, test_class_labels, test_color_labels = read_dataset(ROOT_FOLDER='./images/', gt_json='./images/gt.json')

    #List with all the existant classes
    classes = list(set(list(train_class_labels) + list(test_class_labels)))

    max_data = 100
    images_to_show = 16
    max_k = 7
    knn_k = 3
    color_query = ["Blue", "White"]
    shape_query = "Shirts"

    predicted_color_labels = get_color_predictions(test_imgs, max_k)
    # preds = retrieval_by_color(test_imgs, predicted_color_labels, ["Purple", "Black"] )
    predicted_shape_labels = get_shape_predictions(train_imgs, test_imgs, train_class_labels, knn_k)
    # preds = retrieval_by_shape(test_imgs, predicted_shape_labels, "dresses")

    preds = retrieval_combined(test_imgs, predicted_color_labels, predicted_shape_labels, color_query, shape_query)

    title = "Query for " + "".join(color_query) + " " + shape_query
    names = []
    corrects = []
    for i in range(images_to_show):
        names.append("Predicted: {} {}\nObtained: {} {}".format(
            "".join(predicted_color_labels[i].tolist()), shape_query, "".join(test_color_labels[i]), test_class_labels[i]))

        if set(predicted_color_labels[i].tolist()).issubset(test_color_labels[i]) or set(test_color_labels[i]).issubset(predicted_color_labels[i].tolist()) \
            and predicted_shape_labels[i] == test_class_labels[i]:
            corrects.append(True)
        else:
            corrects.append(False)

    visualize_retrieval(preds, images_to_show, names, corrects, title)

    print(get_color_accuracy(predicted_color_labels, test_color_labels))
    print(get_shape_accuracy(predicted_shape_labels, test_class_labels))
