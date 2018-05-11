#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fake news detection
The TensorFlow version of neural network
"""

import numpy as np
import tensorflow as tf
from getEmbeddings import getEmbeddings
import matplotlib.pyplot as plt
import scikitplot.plotters as skplt
import os.path

IN_DIM = 300
CLASS_NUM = 2
LEARN_RATE = 0.0001
TRAIN_STEP = 20000
tensorflow_tmp = "tmp_tensorflow"


def plot_cmat(yte, ypred):
    '''Plotting confusion matrix'''
    skplt.plot_confusion_matrix(yte,ypred)
    plt.show()


def dummy_input_fn():
    return np.array([1.0] * IN_DIM)


def model_fn(features, labels, mode):
    """The model function for tf.Estimator"""
    # Input layer
    input_layer = tf.reshape(features["x"], [-1, IN_DIM])
    # Dense layer1
    dense1 = tf.layers.dense(inputs=input_layer, units=300, \
        activation=tf.nn.relu)
    # Dropout layer1
    dropout1 = tf.layers.dropout(inputs=dense1, rate=0.4, \
        training=(mode == tf.estimator.ModeKeys.TRAIN))
    # Dense layer2
    dense2 = tf.layers.dense(inputs=dropout1, units=300, \
        activation=tf.nn.relu)
    # Dropout layer2
    dropout2 = tf.layers.dropout(inputs=dense2, rate=0.4, \
        training=(mode == tf.estimator.ModeKeys.TRAIN))
    # Dense layer3
    dense3 = tf.layers.dense(inputs=dropout2, units=300, \
        activation=tf.nn.relu)
    # Dropout layer3
    dropout3 = tf.layers.dropout(inputs=dense3, rate=0.4, \
        training=(mode == tf.estimator.ModeKeys.TRAIN))
    # Logits layer
    logits = tf.layers.dense(inputs=dropout3, units=CLASS_NUM)

    # prediction result in PREDICT and EVAL phases
    predictions = {
        # Class id
        "classes": tf.argmax(input=logits, axis=1),
        # Probabilities
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    
    # Calculate Loss for TRAIN and EVAL
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the training Op
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=LEARN_RATE)
        train_op = optimizer.minimize(\
            loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(\
            mode=mode, loss=loss, train_op=train_op)
    
    # Add evaluation metrics
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(\
            labels=labels, predictions=predictions["classes"])
    }
    return tf.estimator.EstimatorSpec(\
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main():
    # Get the training and testing data from getEmbeddings
    if not os.path.isfile('./xtr.npy') or \
        not os.path.isfile('./xte.npy') or \
        not os.path.isfile('./ytr.npy') or \
        not os.path.isfile('./yte.npy'):
        xtr,xte,ytr,yte = getEmbeddings("datasets/train.csv")
        np.save('./xtr', xtr)
        np.save('./xte', xte)
        np.save('./ytr', ytr)
        np.save('./yte', yte)
    # Read the Doc2Vec data
    train_data = np.load('./xtr.npy')
    eval_data = np.load('./xte.npy')
    train_labels = np.load('./ytr.npy')
    eval_labels = np.load('./yte.npy')
    train_labels = train_labels.reshape((-1, 1)).astype(np.int32)
    eval_labels = eval_labels.reshape((-1, 1)).astype(np.int32)

    # Create the Estimator
    classifier = \
        tf.estimator.Estimator(model_fn=model_fn, model_dir=tensorflow_tmp)

    # Setup logging hook for prediction
    tf.logging.set_verbosity(tf.logging.INFO)
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=200)
    
    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=50,
        num_epochs=None,
        shuffle=True)
    classifier.train(
        input_fn=train_input_fn,
        steps=TRAIN_STEP,
        hooks=[logging_hook])
    
    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
    eval_results = classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)   # 81.42%
    
    # Draw the confusion matrix
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        num_epochs=1,
        shuffle=False)
    predict_results = classifier.predict(input_fn=predict_input_fn)
    predict_labels = [label["classes"] for label in predict_results]
    plot_cmat(eval_labels, predict_labels)


if __name__ == "__main__":
    main()
