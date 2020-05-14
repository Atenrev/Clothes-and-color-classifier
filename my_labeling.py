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


def get_color_predictions(images, k):
    preds = np.empty((len(images), k), dtype='<U8')

    for ix, input in enumerate(images):
        kms = km.KMeans(input, k, {"km_init": "random"})
        kms.fit()
        preds[ix] = km.get_colors(kms.centroids)

    return preds


def get_shape_predictions(train_images, test_images, train_labels, k):
    knn = kn.KNN(train_images, train_labels)
    preds = knn.predict(test_images, k)
    return preds


def retrieval_by_color(images, labels, probs, keywords):
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

    for i in range(color_labels.shape[0]):
        if set(color_keyword).issubset(color_labels[i]) or set(color_labels[i]).issubset(color_keyword):
            mask_color.append(i)

    mask_shape = np.where(shape_keyword == shape_labels)
    idx = np.intersect1d(np.array(mask_color), mask_shape)
    
    return images[idx]


def kmean_statistics(images, kmax):
    for ix, input in enumerate(images):
        kms = km.KMeans(input, 2, {"km_init": "random"})
        local_scores = []
        local_iterations = []


        for k in range(2, kmax+1):
            kms.K = k
            kms.fit()
            score = kms.whitinClassDistance()
            local_scores.append(score)
            local_iterations.append(kms.num_iter)
            print("Results for image " + str(ix) + " with k=" + str(k)) 
            print("Score: " + str(score))
            print("Iterations needed: " + str(kms.num_iter))
            print("")
            # visualize_k_means(kms, input.shape)
        
        score_series = pd.Series(local_scores, index=list(range(2,kmax+1)), name="Score")
        score_series.plot(legend=True)
        iterations_series = pd.Series(local_iterations, index=list(range(2,kmax+1)), name="Iterations")
        iterations_series.plot(legend=True)
        plt.show()


def knn_statistics(train_images, test_images, train_labels, test_labels, kmax):
    kmax += 1
    acc_list = []

    for k in range(2, kmax):
        preds = get_shape_predictions(train_images, test_images, train_labels, k)
        acc_list.append(get_shape_accuracy(preds, test_labels))

    acc_series = pd.Series(acc_list, index=list(range(2,kmax)), name="Accuracy")
    acc_series.plot(legend=True)
    plt.show()


def get_shape_accuracy(predicted, ground_truth):
    return np.sum(predicted == ground_truth) / predicted.shape[0] * 100


def get_color_accuracy(predicted, ground_truth):
    sumup = 0

    for i in range(predicted.shape[0]):
        if set(predicted[i].tolist()).issubset(ground_truth[i]) or set(ground_truth[i]).issubset(predicted[i].tolist()):
            sumup += 1

    return sumup / predicted.shape[0] * 100


if __name__ == '__main__':

    #Load all the images and GT
    train_imgs, train_class_labels, train_color_labels, \
    test_imgs, test_class_labels, test_color_labels = read_dataset(ROOT_FOLDER='./images/', gt_json='./images/gt.json')

    #List with all the existant classes
    classes = list(set(list(train_class_labels) + list(test_class_labels)))

    
    predicted_color_labels = get_color_predictions(test_imgs, 2)
    # preds = retrieval_by_color(test_imgs, predicted_color_labels, predicted_color_probs, ["Purple", "Black"] )
    # predicted_shape_labels = get_shape_predictions(train_imgs, test_imgs, train_class_labels)
    # preds = retrieval_by_shape(test_imgs, predicted_shape_labels, "dresses")
    # preds = retrieval_combined(test_imgs, predicted_color_labels, predicted_shape_labels, ["Blue", "White"], "Socks")
    # visualize_retrieval(preds, 10)

    # print(get_color_accuracy(predicted_color_labels, test_color_labels))

    kmean_statistics(train_imgs, 10)
    # knn_statistics(train_imgs, test_imgs, train_class_labels, test_class_labels, 10)
