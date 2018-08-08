# Copyright 2018 The TensorFlow Probability Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Trains a deep Bayesian neural net to classify MNIST digits."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings
warnings.filterwarnings("ignore")

import pdb
import os
import h5py
from sklearn.preprocessing import MinMaxScaler

# Dependency imports
from absl import flags
import matplotlib
matplotlib.use("Agg")
from matplotlib import figure  # pylint: disable=g-import-not-at-top
from matplotlib.backends import backend_agg
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import matplotlib.pyplot as plt

from tensorflow.contrib.learn.python.learn.datasets import mnist

# TODO(b/78137893): Integration tests currently fail with seaborn imports.
try:
  import seaborn as sns  # pylint: disable=g-import-not-at-top
  HAS_SEABORN = True
except ImportError:
  HAS_SEABORN = False

tfd = tf.contrib.distributions

IMAGE_SHAPE = [28, 28]

flags.DEFINE_float("learning_rate",
                   default=0.00001,
                   help="Initial learning rate.")
flags.DEFINE_integer("max_steps",
                     default=20000,
                     help="Number of training steps to run.")
flags.DEFINE_list("layer_sizes",
                  default=["128", "128"],
                  help="Comma-separated list denoting hidden units per layer.")
flags.DEFINE_string("activation",
                    default="relu",
                    help="Activation function for all hidden layers.")
flags.DEFINE_integer("batch_size",
                     default=100,
                     help="Batch size.")
flags.DEFINE_string("data_dir",
                    default=os.path.join(os.getenv("TEST_TMPDIR", "."),
                                         "bayesian_neural_network/data"),
                    help="Directory where data is stored (if using real data).")
flags.DEFINE_string(
    "model_dir",
    default=os.path.join(os.getenv("TEST_TMPDIR", "."),
                         "bayesian_neural_network/"),
    help="Directory to put the model's fit.")
flags.DEFINE_integer("viz_steps",
                     default=100,
                     help="Frequency at which save visualizations.")
flags.DEFINE_integer("num_monte_carlo",
                     default=200,
                     help="Network draws to compute predictive probabilities.")
flags.DEFINE_bool("fake_data",
                  default=None,
                  help="If true, uses fake data. Defaults to real data.")

FLAGS = flags.FLAGS


def plot_weight_posteriors(names, qm_vals, qs_vals, fname):
  """Save a PNG plot with histograms of weight means and stddevs.
  Args:
    names: A Python `iterable` of `str` variable names.
    qm_vals: A Python `iterable`, the same length as `names`,
      whose elements are Numpy `array`s, of any shape, containing
      posterior means of weight varibles.
    qs_vals: A Python `iterable`, the same length as `names`,
      whose elements are Numpy `array`s, of any shape, containing
      posterior standard deviations of weight varibles.
    fname: Python `str` filename to save the plot to.
  """
  fig = figure.Figure(figsize=(6, 3))
  canvas = backend_agg.FigureCanvasAgg(fig)

  ax = fig.add_subplot(1, 2, 1)
  for n, qm in zip(names, qm_vals):
    sns.distplot(qm.flatten(), ax=ax, label=n)
  ax.set_title("weight means")
  ax.set_xlim([-1.5, 1.5])
  ax.set_ylim([0, 4.])
  ax.legend()

  ax = fig.add_subplot(1, 2, 2)
  for n, qs in zip(names, qs_vals):
    sns.distplot(qs.flatten(), ax=ax)
  ax.set_title("weight stddevs")
  ax.set_xlim([0, 1.])
  ax.set_ylim([0, 25.])

  fig.tight_layout()
  canvas.print_figure(fname, format="png")
  print("saved {}".format(fname))


def plot_heldout_prediction(input_vals, probs,
                            fname, n=10, title=""):
  """Save a PNG plot visualizing posterior uncertainty on heldout data.
  Args:
    input_vals: A `float`-like Numpy `array` of shape
      `[num_heldout] + IMAGE_SHAPE`, containing heldout input images.
    probs: A `float`-like Numpy array of shape `[num_monte_carlo,
      num_heldout, num_classes]` containing Monte Carlo samples of
      class probabilities for each heldout sample.
    fname: Python `str` filename to save the plot to.
    n: Python `int` number of datapoints to vizualize.
    title: Python `str` title for the plot.
  """
  fig, ax = plt.subplots(n, 3)
  fig.set_size_inches(12,30)
  ax = ax.ravel()
  #canvas = backend_agg.FigureCanvasAgg(fig)
  for i in range(n):
    for j in range(3):
      ax[3*i+j].axvline(input_vals[i][j], lw=3)
      ax[3*i+j].hist(probs[:,i,j], bins=30, alpha=0.3, normed=True)

      #ax[3*i+j].set_ylim([0, 1])
      #ax[3*i+j].set_title("posterior samples")

  fig.suptitle(title)
  fig.tight_layout()

  fig.savefig(fname)
  print("saved {}".format(fname))


def plot_line_plot(input_vals, probs,
                            fname, title=""):
  """Save a PNG plot visualizing posterior uncertainty on heldout data.
  Args:
    input_vals: A `float`-like Numpy `array` of shape
      `[num_heldout] + IMAGE_SHAPE`, containing heldout input images.
    probs: A `float`-like Numpy array of shape `[num_monte_carlo,
      num_heldout, num_classes]` containing Monte Carlo samples of
      class probabilities for each heldout sample.
    fname: Python `str` filename to save the plot to.
    n: Python `int` number of datapoints to vizualize.
    title: Python `str` title for the plot.
  """
  fig, ax = plt.subplots(1, 3)
  fig.set_size_inches(10,15)
  ax = ax.ravel()
  means = np.zeros((len(input_vals),3))
  stds = np.zeros((len(input_vals),3))
  #canvas = backend_agg.FigureCanvasAgg(fig)
  for i in range(len(input_vals)):
    for j in range(3): 
      means[i][j] = np.mean(probs[:,i,j])
      stds[i][j] = np.std(probs[:,i,j])
      #ax[3*i+j].set_ylim([0, 1])
      #ax[3*i+j].set_title("posterior samples")
  #pdb.set_trace()
  for j in range(3):
    ax[j].plot(input_vals[:,j], input_vals[:,j],color='k')
    ax[j].errorbar(input_vals[:,j], means[:,j], yerr=stds[:,j])
  fig.suptitle(title)
  fig.tight_layout()

  fig.savefig(fname)
  print("saved {}".format(fname))
  return
def build_input_pipeline(train_vals, train_labels, test_vals, test_labels, batch_size, heldout_size):
  """Build an Iterator switching between train and heldout data."""
  # Build an iterator over training batches.

  #this will waste memory with numpy arrays 
  training_dataset = tf.data.Dataset.from_tensor_slices(
      (train_vals[:,:,np.newaxis], train_labels))
  training_batches = training_dataset.repeat().batch(batch_size)
  training_iterator = training_batches.make_one_shot_iterator()

  # Build a iterator over the heldout set with batch_size=heldout_size,
  # i.e., return the entire heldout set as a constant.
  heldout_dataset = tf.data.Dataset.from_tensor_slices(
      (test_vals[:,:,np.newaxis], test_labels))
  heldout_frozen = (heldout_dataset.take(heldout_size).
                    repeat().batch(heldout_size))
  heldout_iterator = heldout_frozen.make_one_shot_iterator()

  # Combine these into a feedable iterator that can switch between training
  # and validation inputs.
  handle = tf.placeholder(tf.string, shape=[])
  feedable_iterator = tf.data.Iterator.from_string_handle(
      handle, training_batches.output_types, training_batches.output_shapes)
  images, labels = feedable_iterator.get_next()

  return images, labels, handle, training_iterator, heldout_iterator



def main(argv):
  del argv  # unused
  #FLAGS.layer_sizes = [int(units) for units in FLAGS.layer_sizes]
  #FLAGS.activation = getattr(tf.nn, FLAGS.activation)
  output_list = []

  """
  if tf.gfile.Exists(FLAGS.model_dir):
    tf.logging.warning(
        "Warning: deleting old log directory at {}".format(FLAGS.model_dir))
    tf.gfile.DeleteRecursively(FLAGS.model_dir)
  tf.gfile.MakeDirs(FLAGS.model_dir)
  """


  #if FLAGS.fake_data:
  #  mnist_data = build_fake_data()
  #else:
  #  mnist_data = mnist.read_data_sets(FLAGS.data_dir)
  scalerX = MinMaxScaler(feature_range=(-1,1))
  scalerY = MinMaxScaler(feature_range=(-1,1))

  scalerX2 = MinMaxScaler(feature_range=(-1,1))
  with h5py.File('NN_NK_combined_train_set_confined_freq_domain.hdf5', 'r') as f:
    input_data = scalerX.fit_transform(np.asarray(f['amp'],dtype=np.float32).T).T
    input_labels = scalerY.fit_transform(np.asarray(f['fundamental_freqs'],dtype=np.float32))

  with h5py.File('NN_NK_combined_test_set_confined_freq_domain.hdf5', 'r') as f:
    test_data = scalerX2.fit_transform(np.asarray(f['amp'],dtype=np.float32).T).T
    test_labels = scalerY.transform(np.asarray(f['fundamental_freqs'],dtype=np.float32))

  with tf.Graph().as_default():
    (images, labels, handle,
     training_iterator, heldout_iterator) = build_input_pipeline(
         input_data, input_labels, test_data, test_labels, FLAGS.batch_size, test_labels.shape[0])

    # Build a Bayesian neural net. We use the Flipout Monte Carlo estimator for
    # each layer: this enables lower variance stochastic gradients than naive
    # reparameterization.
    with tf.name_scope("bayesian_neural_net", values=[images]):
      neural_net = tf.keras.Sequential()
      #for units in FLAGS.layer_sizes:
      #  layer = tfp.layers.DenseFlipout(
      #      units,
      #      activation=FLAGS.activation)
      #  neural_net.add(layer)
      
      neural_net.add(tfp.layers.Convolution1DFlipout(32, kernel_size=50, activation='relu'))
      neural_net.add(tf.layers.MaxPooling1D(2, 1, padding='same'))
      neural_net.add(tfp.layers.Convolution1DFlipout(32, kernel_size=50, activation='relu'))
      neural_net.add(tf.layers.MaxPooling1D(2, 1, padding='same'))
      neural_net.add(tf.layers.Flatten())
      neural_net.add(tfp.layers.DenseFlipout(1024, activation='relu'))

      #### ADD DROPOUT OR REGULARIZATION

      neural_net.add(tfp.layers.DenseFlipout(3, activation=None))
      predictions = neural_net(images)
      
      """
      neural_net.add(tf.layers.Conv1D(32, kernel_size=50, activation='relu'))
      neural_net.add(tf.layers.MaxPooling1D(2, 1, padding='same'))
      neural_net.add(tf.layers.Conv1D(32, kernel_size=50, activation='relu'))
      neural_net.add(tf.layers.MaxPooling1D(2, 1, padding='same'))
      neural_net.add(tf.layers.Flatten())
      neural_net.add(tf.layers.Dense(1024, activation='relu'))
      neural_net.add(tf.layers.Dense(3, activation='relu'))
      predictions = neural_net(images)
      """
      
      #sess2 = tf.Session()
      """
      for i in range(100):
        neural_net.build()
        predictions2.append(neural_net.call(images))

      
      test = tf.transpose(predictions2, perm=[1,2,0])
      mean, var = tf.nn.moments(test, axes=[-1])

      dists = tf.distributions.Normal(mean, tf.sqrt(var))

      lnpdf = dists.log_prob(labels)
      """

      #changed to uniform (PRIOR????)
      #labels_distribution = tfd.Normal(predictions, 0.1)
      #andom_norm = tf.random_normal_initializer(mean=predictions, stddev= 0.1)

    # Compute the -ELBO as the loss, averaged over the batch size.
    #neg_log_likelihood = -tf.reduce_mean(labels_distribution.log_prob(labels))
    #get posterior, find the probability of the label in the posterior!!!!!!!
    neg_log_likelihood = tf.log(tf.losses.mean_squared_error(labels, predictions))
    #likelihood = 
    kl = sum(neural_net.losses)/(len(input_data))
    
    #ADD PRIOR 
    elbo_loss = neg_log_likelihood + kl
    #elbo_loss = tf.losses.mean_squared_error(labels, predictions)

    # Build metrics for evaluation. Predictions are formed from a single forward
    # pass of the probabilistic layers. They are cheap but noisy predictions.
    #predictions = tf.argmax(logits, axis=1)
    accuracy, accuracy_update_op = tf.metrics.mean_squared_error(
        labels=labels, predictions=predictions)

    # Extract weight posterior statistics for later visualization.
    
    names = []
    qmeans = []
    qstds = []
    for i, layer in enumerate(neural_net.layers):
      if layer.name[0:3] == 'max' or layer.name[0:4] == 'flat':
        continue
      q = layer.kernel_posterior
      names.append("Layer {}".format(i))
      qmeans.append(q.mean())
      qstds.append(q.stddev())
    
    with tf.name_scope("train"):
      opt = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

      train_op = opt.minimize(elbo_loss)
      sess = tf.Session()
      sess.run(tf.global_variables_initializer())
      sess.run(tf.local_variables_initializer())

      # Run the training loop.
      train_handle = sess.run(training_iterator.string_handle())
      heldout_handle = sess.run(heldout_iterator.string_handle())
      for step in range(FLAGS.max_steps):
        _ = sess.run([train_op, accuracy_update_op],
                     feed_dict={handle: train_handle})
        print('Step',step)

        #np.savetxt('check_data_step_{}.txt'.format(step), labels_train)

        
        if step % 20 == 0:
          loss_value, accuracy_value = sess.run(
              [elbo_loss, accuracy], feed_dict={handle: train_handle})
          print("Step: {:>3d} Loss: {:.3f} Accuracy: {:.3f}".format(
              step, loss_value, accuracy_value))

          output_list.append([step, loss_value, accuracy_value])

          np.savetxt('learning_data_mse.txt', np.asarray(output_list), header='step\tloss_value\taccuracy_value')
         
        #if step % 100 == 0:
        #  pdb.set_trace() 

        if (step+1) % 100 == 0:#FLAGS.viz_steps == 0:
          # Compute log prob of heldout set by averaging draws from the model:
          # p(heldout | train) = int_model p(heldout|model) p(model|train)
          #                   ~= 1/n * sum_{i=1}^n p(heldout | model_i)
          # where model_i is a draw from the posterior p(model|train).
          
          probs = np.asarray([sess.run(predictions,
                                       feed_dict={handle: heldout_handle})
                              for _ in range(100)])
          #pdb.set_trace() 
          #mean_probs = np.mean(probs, axis=0)

          image_vals, label_vals = sess.run((images, labels),
                                            feed_dict={handle: heldout_handle})
          
          #heldout_lp = np.mean(np.log(mean_probs[np.arange(mean_probs.shape[0]),label_vals.flatten()]))
          #print(" ... Held-out nats: {:.3f}".format(heldout_lp))

          #qm_vals, qs_vals = sess.run((qmeans, qstds))
          """
          fig, ax = plt.subplots(3, 1)
          ax = ax.ravel()
          for i in range(3):
            ax[i].plot(label_vals[:,0], preds[:,0])
          fig.savefig('checker_{}.png'.format(step))
          """

          if HAS_SEABORN:
            print('start plots')
            """
            plot_weight_posteriors(names, qm_vals, qs_vals,
                                   fname=os.path.join(
                                       FLAGS.model_dir,
                                       "step{:05d}_weights_shuffle.png".format(step)))
            """
            plot_heldout_prediction(label_vals, probs,
                                    fname=os.path.join(
                                        FLAGS.model_dir,
                                        "step{:05d}_pred_mse.png".format(step)))

            plot_line_plot(label_vals, probs,
                                    fname=os.path.join(
                                        FLAGS.model_dir,
                                        "step{:05d}_line_mse.png".format(step)))


if __name__ == "__main__":
  tf.app.run()
