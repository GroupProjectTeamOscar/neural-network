from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import pandas as pd
import tensorflow as tf
import numpy as np

tf.logging.set_verbosity(tf.logging.INFO)

data = pd.read_csv("../../Desktop/deeplearningsociety/code/data/substanceabuse_train.csv")
df=data.loc[1:, :]

COLUMNS = ['%s_'%k for k in range(0,10004)]
COLUMNS.append('label')
df.columns = COLUMNS
training_set = df.loc[:10000]
test_set = df.loc[10000:]

FEATURES = training_set.columns.drop(['label'])
LABEL = "label"


feature_cols = [tf.contrib.layers.real_valued_column(k)
                  for k in FEATURES]

classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_cols,
                                          hidden_units=[100, 10, 20],
                                          n_classes=2,
                                          model_dir="/tmp/health_predict_phraselabel")

def input_fn(data_set):
  feature_cols = {k: tf.constant(data_set[k].values, shape=[data_set[k].size, 1])
                  for k in FEATURES}
  labels = tf.constant(data_set[LABEL].values, shape=[data_set[LABEL].size, 1])
  return feature_cols, labels


classifier.fit(input_fn=lambda: input_fn(training_set), steps=2000)

ev = classifier.evaluate(input_fn=lambda: input_fn(test_set), steps=1)
loss_score = ev["loss"]
print("Loss: {0:f}".format(loss_score))