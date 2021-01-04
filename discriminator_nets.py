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
"""Discriminator networks for text data."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging
import sonnet as snt
import tensorflow as tf
from tensorflow.python.ops.gen_nn_ops import max_pool
import utils

class LSTMEmbedDiscNet(snt.Module):
  """An LSTM discriminator that operates on word indexes."""

  def __init__(self,
               feature_sizes,
               vocab_size,
               use_layer_norm,
               trainable_embedding_size,
               dropout,
               pad_token,
               embedding_source=None,
               vocab_file=None,
               name='LSTMEmbedDiscNet'):
    super(LSTMEmbedDiscNet, self).__init__(name=name)
    self._feature_sizes = feature_sizes
    self._vocab_size = vocab_size
    self._use_layer_norm = use_layer_norm
    self._trainable_embedding_size = trainable_embedding_size
    self._embedding_source = embedding_source
    self._vocab_file = vocab_file
    self._dropout = dropout
    self._pad_token = pad_token
    if self._embedding_source:
      assert vocab_file

    if self._embedding_source:
      self.all_embeddings = utils.make_partially_trainable_embeddings(
          self._vocab_file, self._embedding_source, self._vocab_size,
          self._trainable_embedding_size)
    else:
      self.all_embeddings = tf.get_variable(
          'trainable_embedding',
          shape=[self._vocab_size, self._trainable_embedding_size],
          trainable=True)

    _, self._embedding_size = self.all_embeddings.shape.as_list()

    self.input_fcn = snt.Linear(self._feature_sizes[0])    
    
    encoder_cells = []
    for feature_size in self._feature_sizes:
      # as elsewhere, restore layer norm later (use_layer_norm=self._use_layer_norm)
      encoder_cells += [
          snt.LSTM(feature_size)
      ]
    self.encoder_cell = snt.DeepRNN(encoder_cells)

    self.batch_final = snt.BatchApply(snt.Linear(1))


  def __call__(self, sequence, sequence_length, is_training=True):
    """Connect to the graph.

    Args:
      sequence: A [batch_size, max_sequence_length] tensor of int. For example
        the indices of words as sampled by the generator.
      sequence_length: A [batch_size] tensor of int. Length of the sequence.
      is_training: Boolean, False to disable dropout.

    Returns:
      A [batch_size, max_sequence_length, feature_size] tensor of floats. For
      each sequence in the batch, the features should (hopefully) allow to
      distinguish if the value at each timestep is real or generated.
    """
    batch_size, max_sequence_length = sequence.shape.as_list()
    keep_prob = (1.0 - self._dropout) if is_training else 1.0

    input_embeddings = tf.nn.dropout(self.all_embeddings, 1 - keep_prob)
    embeddings = tf.nn.embedding_lookup(input_embeddings, sequence)
    
    embeddings.shape.assert_is_compatible_with(
        [batch_size, max_sequence_length, self._embedding_size])
    
    position_dim = 8
    embeddings_pos = utils.append_position_signal(embeddings, position_dim)

    embeddings_pos = tf.reshape(
        embeddings_pos,
        [batch_size * max_sequence_length, self._embedding_size + position_dim])

    lstm_inputs = self.input_fcn(embeddings_pos)
    lstm_inputs = tf.reshape(
        lstm_inputs, [batch_size, max_sequence_length, self._feature_sizes[0]])
    lstm_inputs.shape.assert_is_compatible_with(
        [batch_size, max_sequence_length, self._feature_sizes[0]])

    initial_state = self.encoder_cell.initial_state(batch_size)

    # TF2/Sonnet2 requires time-step major, so permute
    lstm_inputs = tf.transpose(lstm_inputs, perm=[1, 0, 2])
    hidden_states, _ = snt.dynamic_unroll(
        core=self.encoder_cell,
        input_sequence=lstm_inputs,
        sequence_length=sequence_length,
        initial_state=initial_state,
        swap_memory=True)

    # Then permute back
    hidden_states = tf.transpose(hidden_states, perm=[1, 0, 2])
    hidden_states.shape.assert_is_compatible_with(
        [batch_size, max_sequence_length,
         sum(self._feature_sizes)])
    logits = self.batch_final(hidden_states)
    logits.shape.assert_is_compatible_with([batch_size, max_sequence_length, 1])
    logits_flat = tf.reshape(logits, [batch_size, max_sequence_length])

    # Mask past first PAD symbol
    #
    # Note that we still rely on tf.nn.bidirectional_dynamic_rnn taking
    # into account the sequence_length properly, because otherwise
    # the logits at a given timestep will depend on the inputs for all other
    # timesteps, including the ones that should be masked.
    mask = utils.get_mask_past_symbol(sequence, self._pad_token)
    masked_logits_flat = logits_flat * tf.cast(mask, tf.float32)

    return masked_logits_flat
