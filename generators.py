# Lint as: python3
# Copyright 2019 DeepMind Technologies Limited and Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Generators for text data."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging
import sonnet as snt
import tensorflow as tf
import tensorflow_probability as tfp

import utils

class LSTMGen(snt.Module):
  """A multi-layer LSTM language model.

  Uses tied input/output embedding weights.
  """

  def __init__(self,
               vocab_size,
               feature_sizes,
               max_sequence_length,
               batch_size,
               use_layer_norm,
               trainable_embedding_size,
               input_dropout,
               output_dropout,
               pad_token,
               temperature,
               embedding_source=None,
               vocab_file=None,
               name='lstm_gen'):
    super(LSTMGen, self).__init__(name=name)
    self._feature_sizes = feature_sizes
    self._max_sequence_length = max_sequence_length
    self._vocab_size = vocab_size
    self._batch_size = batch_size
    self._use_layer_norm = use_layer_norm
    self._trainable_embedding_size = trainable_embedding_size
    self._embedding_source = embedding_source
    self._vocab_file = vocab_file
    self._input_dropout = input_dropout
    self._output_dropout = output_dropout
    self._pad_token = pad_token
    self._temperature = temperature # should see about training this

    if self._embedding_source:
      assert vocab_file

    if self._embedding_source:
      self.all_embeddings = utils.make_partially_trainable_embeddings(
          self._vocab_file, self._embedding_source, self._vocab_size,
          self._trainable_embedding_size)
    else:
      self.all_embeddings = tf.get_variable(
          'trainable_embeddings',
          shape=[self._vocab_size, self._trainable_embedding_size],
          trainable=True)

    _, self._embedding_size = self.all_embeddings.shape.as_list()

    init = tf.random_normal_initializer(mean=1., stddev=0.2)
    self.in_proj = tf.Variable(init(shape=(self._embedding_size, self._feature_sizes[0])), name='in_proj')
    self.out_bias = tf.Variable(tf.zeros(shape=(1, self._vocab_size)), name='out_bias', dtype=tf.float32)    
    
    # If more than 1 layer, then output has dim sum(self._feature_sizes),
    # which is different from input dim == self._feature_sizes[0]
    # So we need a different projection matrix for input and output.
    if len(self._feature_sizes) > 1:
      self.out_proj = tf.Variable(tf.ones(shape=(self._embedding_size, sum(self._feature_sizes))), name='out_proj')
    else:
      self.out_proj = self.in_proj

    self.encoder_cells = []
    for feature_size in self._feature_sizes:
      # , use_layer_norm=self._use_layer_norm
      self.encoder_cells += [
          snt.LSTM(hidden_size=feature_size)
      ]
    
    self.encoder_cell = snt.deep_rnn_with_skip_connections(self.encoder_cells)

  def __call__(self, is_training=True, temperature=None, writer=None, step=0):
    input_keep_prob = (1. - self._input_dropout) if is_training else 1.0
    output_keep_prob = (1. - self._output_dropout) if is_training else 1.0

    batch_size = self._batch_size
    max_sequence_length = self._max_sequence_length
        
    logging.info(f'Calling generator, max length: {max_sequence_length}, embedding size: {self._embedding_size}, feature sizes: {self._feature_sizes}')

    input_embeddings = tf.nn.dropout(self.all_embeddings, 1 - input_keep_prob)
    output_embeddings = tf.nn.dropout(self.all_embeddings, 1 - output_keep_prob)
    
    state = self.encoder_cell.initial_state(batch_size)
    # logging.info(f'Initial state shape of hidden: {state[0].shape}')

    # Manual unrolling.
    samples_list, logits_list, logprobs_list, embeddings_list = [], [], [], []
    sample = tf.tile(
        tf.constant(self._pad_token, dtype=tf.int32)[None], [batch_size])
    logging.info(f'Sample shape: {sample.shape}, and first row: {sample[0]}')
    for t in range(max_sequence_length):
      # Input is sampled word at t-1.
      if t < 5:
        logging.info(f'Time {t}, provided sample for first sequence in batch: {sample[0]}')
      embedding = tf.nn.embedding_lookup(input_embeddings, sample)
      embedding.shape.assert_is_compatible_with([batch_size, self._embedding_size])
      if t < 5:
        logging.info(f'Time {t}, input embedding, first in sequence: {embedding[0][:10]}')
      
      embedding_proj = tf.matmul(embedding, self.in_proj)
      if t < 5:
        logging.info(f'Embedding, after projection, shape: {embedding_proj[0].shape}, and values: {embedding_proj[0][:10]}')
      embedding_proj.shape.assert_is_compatible_with([batch_size, self._feature_sizes[0]])

      outputs, state = self.encoder_cell(embedding_proj, state)
      # outputs.shape.assert_is_compatible_with([self._embedding_size, sum(self._feature_sizes)])
      if t < 5:
        logging.info(f'Outputs from encoder cell: {outputs[0][:10]}, max: {max(outputs[0])}, at index: {tf.math.argmax(outputs[0])}')
        # logging.info(f'Current state, shape of hidden: {state.hidden.shape}')

      logging.debug(f'Now output shape: {outputs.shape} and output projection shape: {self.out_proj.shape}')
      outputs_proj = tf.matmul(outputs, self.out_proj, transpose_b=True)
      logits = tf.matmul(
          outputs_proj, output_embeddings, transpose_b=True) + self.out_bias
      if t < 5:
        logging.info(f'Logits in generator: {logits[0][:10]}')

      sample_temperature = temperature if temperature else self._temperature
      categorical = tfp.distributions.Categorical(logits=logits/sample_temperature)
      sample = categorical.sample()
      if t < 5:
        logging.info(f'At time step {t}, sampled token: {sample[0]}, logit for token: {logits[0][sample[0]]}, max logit: {max(logits[0])}')
      logprobs = categorical.log_prob(sample)

      samples_list.append(sample)
      logits_list.append(logits)
      logprobs_list.append(logprobs)
      embeddings_list.append(embedding)

    # Create an op to retrieve embeddings for full sequence, useful for testing.
    embeddings = tf.stack(  # pylint: disable=unused-variable
        embeddings_list,
        axis=1,
        name='embeddings')
    sequence = tf.stack(samples_list, axis=1)
    logprobs = tf.stack(logprobs_list, axis=1)

    # The sequence stops after the first occurrence of a PAD token.
    sequence_length = utils.get_first_occurrence_indices(
        sequence, self._pad_token)
    mask = utils.get_mask_past_symbol(sequence, self._pad_token)
    masked_sequence = sequence * tf.cast(mask, tf.int32)
    masked_logprobs = logprobs * tf.cast(mask, tf.float32)

    if writer:
      with writer.as_default():
        logging.info(f'Passed a writer, recording weights, step: {step}')
        tf.summary.histogram('weight/input_proj', self.in_proj, step=step)
        # tf.summary.histogram('gen/first_lstm', self.)
        tf.summary.histogram('weight/output_proj', self.out_proj, step=step)

    return {
        'sequence': masked_sequence,
        'sequence_length': sequence_length,
        'logprobs': masked_logprobs
    }
