from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import pandas as pd
import tensorflow as tf
import numpy as np

tf.logging.set_verbosity(tf.logging.INFO)

data = pd.read_csv("SubstanceAbuseOutput.csv")
df=data.loc[1:, :]
df['label'] = np.where(((df['hour']>21)|(df['hour']<6)),1,0)
df=df.drop(['hour'], axis=1)

COLUMNS = ['%s_'%k for k in range(0,10004)]
COLUMNS.append('label')
df.columns = COLUMNS
training_set = df.loc[:2000]
test_set = df.loc[2000:]

FEATURES = training_set.columns.drop(['0_','label'],axis=1)
LABEL = "label"


feature_cols = [tf.contrib.layers.real_valued_column(k)
                  for k in FEATURES]

classifier = tf.contrib.learn.DNNRegressor(feature_columns=feature_cols,
                                          hidden_units=[10, 10],
                                          n_classes=2,
                                          model_dir="/tmp/substance_abuse_predict_night")

def input_fn(data_set):
  feature_cols = {k: tf.constant(data_set[k].values, shape=[data_set[k].size, 1])
                  for k in FEATURES}
  labels = tf.constant(data_set[LABEL].values, shape=[data_set[LABELatom].size, 1])
  return feature_cols, labels


classifier.fit(input_fn=lambda: input_fn(training_set), steps=5000)

ev = classifier.evaluate(input_fn=lambda: input_fn(test_set), steps=1)
loss_score = ev["loss"]
print("Loss: {0:f}".format(loss_score))