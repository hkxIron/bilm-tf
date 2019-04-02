'''
Train and test bidirectional language models.
'''

import os
import time
import json
import re

import tensorflow as tf
import numpy as np

from tensorflow.python.ops.init_ops import glorot_uniform_initializer

from .data import Vocabulary, UnicodeCharsVocabulary, InvalidNumberOfCharacters

DTYPE = 'float32'
DTYPE_INT = 'int64'

tf.logging.set_verbosity(tf.logging.INFO)

# 这里可以获取所有的variable以及shape
def print_variable_summary():
    import pprint
    variables = sorted([[v.name, v.get_shape()] for v in tf.global_variables()])
    pprint.pprint(variables)

class LanguageModel(object):
    '''
    A class to build the tensorflow computational graph for NLMs

    All hyperparameters and model configuration is specified in a dictionary
    of 'options'.

    is_training is a boolean used to control behavior of dropout layers
        and softmax.  Set to False for testing.

    The LSTM cell is controlled by the 'lstm' key in options
    Here is an example:

     'lstm': {
      'cell_clip': 5,
      'dim': 4096,
      'n_layers': 2,
      'proj_clip': 5,
      'projection_dim': 512,
      'use_skip_connections': True},

        'projection_dim' is assumed token embedding size and LSTM output size.
        'dim' is the hidden state size.
        Set 'dim' == 'projection_dim' to skip a projection layer.
    '''
    def __init__(self, options, is_training):
        self.options = options
        self.is_training = is_training
        self.bidirectional = options.get('bidirectional', False)

        # use word or char inputs?
        self.char_inputs = 'char_cnn' in self.options

        # for the loss function
        self.share_embedding_softmax = options.get('share_embedding_softmax', False)
        if self.char_inputs and self.share_embedding_softmax:
            raise ValueError("Sharing softmax and embedding weights requires "
                             "word input")
        self.sample_softmax = options.get('sample_softmax', True)
        self._build()

    def _build_word_embeddings(self):
        n_tokens_vocab = self.options['n_tokens_vocab']
        batch_size = self.options['batch_size']
        unroll_steps = self.options['unroll_steps']

        # LSTM options
        projection_dim = self.options['lstm']['projection_dim']

        # the input token_ids and word embeddings
        # token_ids: [batch, unroll_steps]
        self.token_ids = tf.placeholder(DTYPE_INT,
                               shape=(batch_size, unroll_steps),
                               name='token_ids')
        # the word embeddings
        with tf.device("/cpu:0"):
            # self.embedding_weights: [vocab_size, projection_dim]
            self.embedding_weights = tf.get_variable(
                "embedding", [n_tokens_vocab, projection_dim],
                dtype=DTYPE,
            )
            # self.embedding: [batch, unroll_steps, projection_dim]
            self.embedding = tf.nn.embedding_lookup(self.embedding_weights,
                                                self.token_ids)

        # if a bidirectional LM then make placeholders for reverse
        # model and embeddings
        if self.bidirectional:
            self.token_ids_reverse = tf.placeholder(DTYPE_INT,
                               shape=(batch_size, unroll_steps),
                               name='token_ids_reverse')
            with tf.device("/cpu:0"):
                self.embedding_reverse = tf.nn.embedding_lookup(
                    self.embedding_weights, self.token_ids_reverse)

    """
    对于char_embedding,在char_num维度上进行卷积以及max-pooling,注意:此处并没有用到word的信息
    """
    def _build_word_char_embeddings(self):
        '''
        options contains key 'char_cnn': {

        'n_characters': 262,

        # includes the start / end characters
        'max_characters_per_token': 50,

        'filters': [
            [1, 32], # kernel=1,filter_count=32
            [2, 32],
            [3, 64],
            [4, 128],
            [5, 256],
            [6, 512],
            [7, 512]
        ],
        'activation': 'tanh',

        # for the character embedding
        'embedding': {'dim': 16}

        # for highway layers
        # if omitted, then no highway layers
        'n_highway': 2,
        }
        '''
        batch_size = self.options['batch_size']
        unroll_steps = self.options['unroll_steps']
        projection_dim = self.options['lstm']['projection_dim']
    
        cnn_options = self.options['char_cnn']
        filters = cnn_options['filters']
        n_filters = sum(f[1] for f in filters) # filter:[kernel,num]
        max_char_count_in_word = cnn_options['max_characters_per_token']
        char_embed_dim = cnn_options['embedding']['dim']
        n_chars = cnn_options['n_characters']
        if n_chars != 261: # 256 + 5 (sos, eos, unk, pad, mask)
            raise InvalidNumberOfCharacters(
                    "Set n_characters=261 for training see the README.md"
            )
        if cnn_options['activation'] == 'tanh':
            activation = tf.nn.tanh
        elif cnn_options['activation'] == 'relu':
            activation = tf.nn.relu

        # the input character ids
        # tokens_characters:[batch, unroll_steps=word_count_in_seq, max_chars=char_count_in_word]
        self.tokens_characters = tf.placeholder(DTYPE_INT,
                                   shape=(batch_size, unroll_steps, max_char_count_in_word),
                                   name='tokens_characters')
        # the character embeddings
        with tf.device("/cpu:0"): # embedding一般在cpu上比较合适
            # [char_count, char_embed_size]
            self.embedding_weights = tf.get_variable(
                    "char_embed",
                    shape=[n_chars, char_embed_dim],
                    dtype=DTYPE,
                    initializer=tf.random_uniform_initializer(-1.0, 1.0)
            )
            # char_embedding: [batch_size, unroll_steps, max_chars, embed_dim]
            self.char_embedding = tf.nn.embedding_lookup(self.embedding_weights,
                                                    self.tokens_characters)

            if self.bidirectional:
                self.tokens_characters_reverse = tf.placeholder(DTYPE_INT,
                                   shape=(batch_size, unroll_steps, max_char_count_in_word),
                                   name='tokens_characters_reverse')
                # char_embedding_reverse: [batch_size, unroll_steps, max_chars, embed_dim]
                self.char_embedding_reverse = tf.nn.embedding_lookup(
                    self.embedding_weights, self.tokens_characters_reverse)


        # the convolutions
        def make_convolutions(input, reuse):
            with tf.variable_scope('CNN', reuse=reuse) as scope:
                convolutions = []
                # filters:[ [1, 32], ... ] # kernel=1,filter_count=32
                for i, (filter_width, filter_num) in enumerate(filters):
                    if cnn_options['activation'] == 'relu':
                        # He initialization for ReLU activation
                        # with char embeddings init between -1 and 1
                        # w_init = tf.random_normal_initializer(
                        #    mean=0.0,
                        #    stddev=np.sqrt(2.0 / (filter_width * char_embed_dim))
                        #)

                        # Kim et al 2015, +/- 0.05
                        w_init = tf.random_uniform_initializer(minval=-0.05, maxval=0.05)
                    elif cnn_options['activation'] == 'tanh':
                        # glorot init
                        w_init = tf.random_normal_initializer(
                            mean=0.0,
                            stddev=np.sqrt(1.0 / (filter_width * char_embed_dim))
                        )
                    # 注意:每个卷积核有不同的参数
                    # w: [filter_height=1, filter_width=filter_width, in_channels=char_embed_dim, out_channels=filter_num]
                    w = tf.get_variable(
                        "W_cnn_%s" % i,
                        [1, filter_width, char_embed_dim, filter_num],
                        initializer=w_init,
                        dtype=DTYPE)
                    # b:[filter_num]
                    b = tf.get_variable(
                        "b_cnn_%s" % i, [filter_num], dtype=DTYPE,
                        initializer=tf.constant_initializer(0.0))
                    # input:[batch, in_height=unroll_steps, in_width=max_chars, in_channels=embed_dim]
                    # conv:[batch, unroll_steps, (max_chars-kernel_size)//stride+1, filter_num], stride=1
                    # =>   [batch, unroll_steps, max_chars-kernel_size+1, filter_num]
                    conv = tf.nn.conv2d(
                            input=input,
                            filter=w,
                            strides=[1, 1, 1, 1], # [1, height_stride, width_stide, 1]
                            padding="VALID") + b # valid:不填充
                    # now max pool
                    # conv:[batch, unroll_steps, max_chars-kernel_size+1, filter_num]
                    # conv_new:[batch, unroll_steps, 1, filter_num]
                    conv = tf.nn.max_pool(
                            value=conv,
                            ksize=[1, 1, max_char_count_in_word-filter_width+1, 1],
                            strides=[1, 1, 1, 1],
                            padding='VALID')

                    # activation
                    # conv:[batch, unroll_steps, 1, filter_num]
                    conv = activation(conv)
                    # conv:[batch, unroll_steps, filter_num]
                    conv = tf.squeeze(conv, squeeze_dims=[2])

                    # convolutions:[ [batch, unroll_steps, filter_num], [b, u, f], ... ]
                    convolutions.append(conv)

            # convolutions:[batch, unroll_steps, total_filter_num]
            return tf.concat(convolutions, 2)

        # for first model, this is False, for others it's True
        reuse = tf.get_variable_scope().reuse # 这里根据GPU来设置,后面的gpu重用第一个gpu的变量
        """
        对于char_embedding,在char_num维度上进行卷积以及max-pooling
        """
        # char_embedding: [batch_size, unroll_steps, max_chars, embed_dim]
        # embedding:[batch, unroll_steps, total_filter_num]
        embedding = make_convolutions(self.char_embedding, reuse)

        # token_embedding_layers:list([batch, unroll_steps, total_filter_num])
        # 1.加入cnn后的embedding
        self.token_embedding_layers = [embedding]

        if self.bidirectional:
            # re-use the CNN weights from forward pass
            embedding_reverse = make_convolutions(self.char_embedding_reverse, True)

        # for highway and projection layers:
        #   reshape from (batch_size, n_tokens, dim) to
        n_highway = cnn_options.get('n_highway') # 2层highway
        use_highway = n_highway is not None and n_highway > 0
        use_proj = n_filters != projection_dim

        if use_highway or use_proj:
            # embedding:[batch, unroll_steps, total_filter_num]
            #         =>[batch*unroll_steps, total_filter_num]
            embedding = tf.reshape(embedding, [-1, n_filters])
            if self.bidirectional:
                embedding_reverse = tf.reshape(embedding_reverse,
                    [-1, n_filters])

        # set up weights for projection
        if use_proj:
            assert n_filters > projection_dim
            with tf.variable_scope('CNN_proj') as scope:
                    # w:[n_filters, projection_dim]
                    W_proj_cnn = tf.get_variable(
                        "W_proj", [n_filters, projection_dim],
                        initializer=tf.random_normal_initializer(
                            mean=0.0, stddev=np.sqrt(1.0 / n_filters)),
                        dtype=DTYPE)
                    # b:[project_num]
                    b_proj_cnn = tf.get_variable(
                        "b_proj", [projection_dim],
                        initializer=tf.constant_initializer(0.0),
                        dtype=DTYPE)

        # apply highways layers
        def highway(x, ww_carry, bb_carry, ww_tr, bb_tr):
            # x:[N, dim], ww_carry:[dim, dim], bb_carry:[dim]
            # g=f(x*w+b)
            carry_gate = tf.nn.sigmoid(tf.matmul(x, ww_carry) + bb_carry)
            transform_gate = tf.nn.relu(tf.matmul(x, ww_tr) + bb_tr)
            return carry_gate * transform_gate + (1.0 - carry_gate) * x  #此处与residual稍有不同

        if use_highway:
            highway_dim = n_filters

            for i in range(n_highway): # 2层highway
                with tf.variable_scope('CNN_highway_%s' % i) as scope:
                    # w_carry:[highway_dim=n_filters, highway_dim]
                    W_carry = tf.get_variable(
                        'W_carry', [highway_dim, highway_dim],
                        # glorit init
                        initializer=tf.random_normal_initializer(
                            mean=0.0, stddev=np.sqrt(1.0 / highway_dim)),
                        dtype=DTYPE)
                    # b_carry:[highway_dim=n_filters]
                    b_carry = tf.get_variable(
                        'b_carry', [highway_dim],
                        initializer=tf.constant_initializer(-2.0),
                        dtype=DTYPE)
                    # w_transform:[highway_dim=n_filters, highway_dim]
                    W_transform = tf.get_variable(
                        'W_transform', [highway_dim, highway_dim],
                        initializer=tf.random_normal_initializer(
                            mean=0.0, stddev=np.sqrt(1.0 / highway_dim)),
                        dtype=DTYPE)
                    b_transform = tf.get_variable(
                        'b_transform', [highway_dim],
                        initializer=tf.constant_initializer(0.0),
                        dtype=DTYPE)

                # embedding:[batch*unroll_steps, total_filter_num=n_filters]
                # => [batch*unroll_steps, total_filter_num]
                embedding = highway(embedding, W_carry, b_carry,
                                    W_transform, b_transform)

                if self.bidirectional:
                    embedding_reverse = highway(embedding_reverse,
                                                W_carry, b_carry,
                                                W_transform, b_transform)
                # 2.加入highway后的embedding
                self.token_embedding_layers.append(
                    tf.reshape(embedding, 
                        [batch_size, unroll_steps, highway_dim])
                )

        # finally project down to projection dim if needed
        if use_proj:
            # embedding:[batch*unroll_steps, total_filter_num]
            # => [batch*unroll_steps, projection_dim]
            embedding = tf.matmul(embedding, W_proj_cnn) + b_proj_cnn
            if self.bidirectional:
                embedding_reverse = tf.matmul(embedding_reverse, W_proj_cnn) \
                    + b_proj_cnn
            # 3.加入projection后的embedding
            self.token_embedding_layers.append(
                tf.reshape(embedding,
                        [batch_size, unroll_steps, projection_dim])
            )

        # reshape back to (batch_size, tokens, dim)
        if use_highway or use_proj:
            shp = [batch_size, unroll_steps, projection_dim]
            # embedding:[batch, unroll_steps, projection_dim]
            embedding = tf.reshape(embedding, shp)
            if self.bidirectional:
                embedding_reverse = tf.reshape(embedding_reverse, shp)

        # embedding:[batch, unroll_steps, projection_dim]
        # at last assign attributes for remainder of the model
        self.embedding = embedding
        if self.bidirectional:
            self.embedding_reverse = embedding_reverse

    def _get_lstm_output(self, batch_size):
        # now the LSTMs
        # these will collect the initial states for the forward
        #   (and reverse LSTMs if we are doing bidirectional)
        self.init_lstm_state = []
        self.final_lstm_state = []

        # LSTM options
        lstm_dim = self.options['lstm']['dim']
        projection_dim = self.options['lstm']['projection_dim']
        n_lstm_layers = self.options['lstm'].get('n_layers', 1)
        dropout = self.options['dropout']
        keep_prob = 1.0 - dropout

        # get the LSTM inputs
        if self.bidirectional:
            # self.embedding:[batch, unroll_steps, projection_dim]
            lstm_inputs = [self.embedding, self.embedding_reverse]
        else:
            lstm_inputs = [self.embedding]

        # now compute the LSTM outputs
        cell_clip = self.options['lstm'].get('cell_clip')
        proj_clip = self.options['lstm'].get('proj_clip')

        use_skip_connections = self.options['lstm'].get('use_skip_connections')
        if use_skip_connections:
            print("USING SKIP CONNECTIONS")
        else:
            print("NOT USING SKIP CONNECTIONS")

        """
        lstm_units=30
        x=20 
        gate_num=4, input_gate, output_gate, forget_gate, candidate_gate
        bias =120 =30*4 
        kernel= x* 4
         
        ['lm/RNN_0/rnn/multi_rnn_cell/cell_0/lstm_cell/bias:0',  TensorShape([Dimension(120)])], 120=30*4
        ['lm/RNN_0/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel:0',  TensorShape([Dimension(20),Dimension(120)])],
        ['lm/RNN_0/rnn/multi_rnn_cell/cell_0/lstm_cell/projection/kernel:0',  TensorShape([Dimension(30),Dimension(10)])],
        
        ['lm/RNN_0/rnn/multi_rnn_cell/cell_1/lstm_cell/bias:0',  TensorShape([Dimension(120)])],
        ['lm/RNN_0/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel:0',  TensorShape([Dimension(20),Dimension(120)])],
        ['lm/RNN_0/rnn/multi_rnn_cell/cell_1/lstm_cell/projection/kernel:0',  TensorShape([Dimension(30),Dimension(10)])],
        """
        lstm_outputs = []
        # lstm_inputs = [self.embedding, self.embedding_reverse]
        for lstm_num, lstm_input in enumerate(lstm_inputs): # 在正向与反向上遍历
            lstm_cells = []
            for i in range(n_lstm_layers): # lstm stacked多少层,原始paper中只有两层
                if projection_dim < lstm_dim: # project_dim:512, lstm_dim:4096
                    # are projecting down output
                    lstm_cell = tf.nn.rnn_cell.LSTMCell(
                        num_units=lstm_dim, # 30
                        num_proj=projection_dim, # 10
                        cell_clip=cell_clip,
                        proj_clip=proj_clip)
                else:
                    lstm_cell = tf.nn.rnn_cell.LSTMCell(
                        num_units=lstm_dim,
                        cell_clip=cell_clip,
                        proj_clip=proj_clip)

                if use_skip_connections:
                    # ResidualWrapper adds inputs to outputs
                    if i == 0:
                        # don't add skip connection from token embedding to
                        # 1st layer output
                        pass
                    else:
                        # add a skip connection
                        lstm_cell = tf.nn.rnn_cell.ResidualWrapper(lstm_cell) # skip connection

                # add dropout, 一般的情况下应该是将test阶段设置keep_prob=1
                if self.is_training:
                    lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell,
                                                              input_keep_prob=keep_prob)
                # 两层lstm
                lstm_cells.append(lstm_cell)

            if n_lstm_layers > 1:
                lstm_cell = tf.nn.rnn_cell.MultiRNNCell(cells=lstm_cells) # 多层lstm
            else:
                lstm_cell = lstm_cells[0]

            """
            tf.nn.static_rnn 和 tf.contrib.rnn.static_rnn 是一样的,都表示同一个
            
            """
            # lstm_input:[embedding:[batch, unroll_steps, projection_dim],
            # embedding_reverse:[batch, unroll_steps, projection_dim]]
            with tf.control_dependencies([lstm_input]):
                self.init_lstm_state.append(lstm_cell.zero_state(batch_size, DTYPE))
                # NOTE: this variable scope is for backward compatibility
                # with existing models...
                if self.bidirectional:
                    # ['lm/RNN_0/rnn/multi_rnn_cell/cell_0/lstm_cell/bias:0',  TensorShape([Dimension(120)])],
                    # ['lm/RNN_0/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel:0',  TensorShape([Dimension(20),Dimension(120)])]
                    with tf.variable_scope('RNN_%s' % lstm_num): # 正向与反向
                        # _lstm_output_unpacked:list, size:time_step* [batch, hidden_size]
                        # final_state: Tuple(h:[batch, hidden_size], c:[batch, hidden_size])
                        _lstm_output_unpacked, final_state = tf.nn.static_rnn(
                            cell=lstm_cell,
                            inputs=tf.unstack(lstm_input, axis=1), # 按时间步 unroll_steps展开
                            initial_state=self.init_lstm_state[-1]) # 利用最近的一个state初始化
                else:
                    _lstm_output_unpacked, final_state = tf.nn.static_rnn(
                        lstm_cell,
                        tf.unstack(lstm_input, axis=1),
                        initial_state=self.init_lstm_state[-1])

                # final_lstm_state:list(Tuple(h:[batch, hidden_size], c:[batch, hidden_size]))
                self.final_lstm_state.append(final_state)

            # _lstm_output_unpacked:list, size:unroll_steps* [batch, hidden_size=project_dim]
            # lstm_output_flat: (batch_size * unroll_steps, projection_dim=512)
            lstm_output_flat = tf.reshape(tf.stack(_lstm_output_unpacked, axis=1), shape=[-1, projection_dim])
            if self.is_training:
                # add dropout to output
                lstm_output_flat = tf.nn.dropout(lstm_output_flat, keep_prob)
            tf.add_to_collection('lstm_output_embeddings', _lstm_output_unpacked)

            # lstm_outputs: list of [batch_size * unroll_steps, projection_dim=512], len为1或2
            lstm_outputs.append(lstm_output_flat)

        # lstm_outputs: list of [batch_size * unroll_steps, projection_dim=512], len为1或2
        return lstm_outputs

    """
    注意: emlo中char embedding以及word embedding没用同时使用,但个人感觉,word+char可以同时利用
    1-char. char embeding:
        a. char-embeding
        b. conv-max-pooling layer
        c. 2-highway-layer
    1-word. word embedding:
        a. token-embeding 
    
    2. bi-lstm
    3. cross-entropy loss
    """
    def _build(self):
        # size of input options
        n_tokens_vocab = self.options['n_tokens_vocab']
        batch_size = self.options['batch_size']
        unroll_steps = self.options['unroll_steps']

        dropout = self.options['dropout']
        keep_prob = 1.0 - dropout

        if self.char_inputs:
            # self.embedding:[batch, unroll_steps, projection_dim]
            self._build_word_char_embeddings()
        else:
            # self.embedding:[batch, unroll_steps, projection_dim]
            self._build_word_embeddings()

        # lstm_outputs: list of [batch_size * unroll_steps, projection_dim=512], len为1或2
        lstm_outputs = self._get_lstm_output(batch_size)
        # 注意:language model里,需要将所有time_step的hidden_state输出
        self._build_loss(lstm_outputs)

    def _build_loss(self, lstm_outputs):
        '''
        lstm_outputs: list of [batch_size * unroll_steps, projection_dim=512], len为1或2

        Create:
            self.total_loss: total loss op for training
            self.softmax_W, softmax_b: the softmax variables
            self.next_token_id / _reverse: placeholders for gold input

        '''
        batch_size = self.options['batch_size']
        unroll_steps = self.options['unroll_steps']
        n_tokens_vocab = self.options['n_tokens_vocab']

        # DEFINE next_token_id and *_reverse placeholders for the gold input
        def _get_next_token_placeholders(suffix):
            name = 'next_token_id' + suffix
            id_placeholder = tf.placeholder(DTYPE_INT,
                                   shape=(batch_size, unroll_steps),
                                   name=name)
            return id_placeholder

        # get the window and weight placeholders
        self.next_token_id = _get_next_token_placeholders('')
        if self.bidirectional:
            self.next_token_id_reverse = _get_next_token_placeholders('_reverse')

        # DEFINE THE SOFTMAX VARIABLES
        # get the dimension of the softmax weights
        # softmax dimension is the size of the output projection_dim
        softmax_dim = self.options['lstm']['projection_dim']

        # the output softmax variables -- they are shared if bidirectional
        if self.share_embedding_softmax: # 共享embedding, 就是原来的embedding矩阵
            # softmax_W is just the embedding layer
            self.softmax_W = self.embedding_weights

        with tf.variable_scope('softmax'), tf.device('/cpu:0'):
            # Glorit init (std=(1.0 / sqrt(fan_in)), 注意:乘的时候会进行转置, 因此fan_in = softmax_dim
            softmax_init = tf.random_normal_initializer(0.0, 1.0 / np.sqrt(softmax_dim))
            if not self.share_embedding_softmax: # 如果不共享embedding
                # softmax_W:[n_token_vacab, project_dim]
                self.softmax_W = tf.get_variable(
                    'W', [n_tokens_vocab, softmax_dim],
                    dtype=DTYPE,
                    initializer=softmax_init
                )
            # softmax_b:[n_token_vacab]
            self.softmax_b = tf.get_variable(
                'b', [n_tokens_vocab], # 注意:此处不是softmax_dim
                dtype=DTYPE,
                initializer=tf.constant_initializer(0.0))

        # now calculate losses
        # loss for each direction of the LSTM
        self.individual_losses = []

        # next_ids:list([batch_size, unroll_steps],), len为1或者2
        if self.bidirectional:
            next_ids = [self.next_token_id, self.next_token_id_reverse]
        else:
            next_ids = [self.next_token_id]

        # next_ids:list([batch_size, unroll_steps],), len为1或者2
        # lstm_outputs: list of [batch_size * unroll_steps, projection_dim=512], len为1或者2
        # list循环,对于正向与反向lstm,会将loss加起来
        for id_placeholder, lstm_output_flat in zip(next_ids, lstm_outputs): # 遍历正向与反向
            # flatten the LSTM output and next token id gold to shape:
            # (batch_size * unroll_steps, softmax_dim)
            # Flatten and reshape the token_id placeholders

            # next_token_id_flat:[batch_size*unroll_steps, 1]
            next_token_id_flat = tf.reshape(id_placeholder, [-1, 1])

            # lstm_outputs: list of [batch_size * unroll_steps, projection_dim=512], len为1或者2
            with tf.control_dependencies([lstm_output_flat]):
                if self.is_training and self.sample_softmax:
                    # softmax_W:[n_token_vacab, project_dim]
                    losses = tf.nn.sampled_softmax_loss(
                                   weights=self.softmax_W,
                                   biases=self.softmax_b,
                                   labels=next_token_id_flat,
                                   inputs=lstm_output_flat,
                                   num_sampled=self.options['n_negative_samples_batch'],
                                   num_classes=self.options['n_tokens_vocab'],
                                   num_true=1)

                else:
                    # get the full softmax loss
                    # lstm_output_flat:[batch_size * unroll_steps, projection_dim=512]
                    # softmax_W:[n_token_vacab, project_dim]
                    # softmax_b:[n_token_vacab]
                    # output_scores:[batch_size * unroll_steps, n_token_vacab]
                    output_scores = tf.matmul(
                        a=lstm_output_flat,
                        b=tf.transpose(self.softmax_W)
                    ) + self.softmax_b
                    # NOTE: tf.nn.sparse_softmax_cross_entropy_with_logits
                    #   expects unnormalized output since it performs the
                    #   softmax internally

                    # output_scores:[batch_size * unroll_steps, n_token_vacab]
                    # next_token_id_flat:[batch_size*unroll_steps, 1]
                    #          squeeze =>[batch_size*unroll_steps]
                    # losses:[batch_size*unroll_steps]
                    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                        logits=output_scores,
                        labels=tf.squeeze(next_token_id_flat, squeeze_dims=[1])
                    )
            # 因此,对于正向与反向lstm,会将loss append起来
            # scalar
            self.individual_losses.append(tf.reduce_mean(losses))

        # now make the total loss -- it's the mean of the individual losses
        # 注意:双向lstm需要平均
        if self.bidirectional:
            self.total_loss = 0.5 * (self.individual_losses[0] + self.individual_losses[1])
        else:
            self.total_loss = self.individual_losses[0]

def average_gradients(tower_grads, batch_size, options):
    """
    # tower_grad_vars:
    # [
    # [(grad0_gpu0, var0_gpu0), (grad1_gpu0, var1_gpu0),.., (gradN_gpu1, varN_gpu1) ],
    # [(grad0_gpu1, var0_gpu1), (grad1_gpu1, var1_gpu1),.., (gradN_gpu2, varN_gpu2) ],
    # ...
    # [(grad0_gpuk, var0_gpuk), (grad1_gpuk, var1_gpuk),.., (gradN_gpuk, varN_gpuk) ],
    # ]
    """
    # calculate average gradient for each shared variable across all GPUs
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        # We need to average the gradients across each GPU.

        g0, v0 = grad_and_vars[0]

        if g0 is None:
            # no gradient for this variable, skip it
            average_grads.append((g0, v0))
            continue

        if isinstance(g0, tf.IndexedSlices): # sparseTensor
            # If the gradient is type IndexedSlices then this is a sparse
            #   gradient with attributes indices and values.
            # To average, need to concat them individually then create
            #   a new IndexedSlices object.
            indices = []
            values = []
            for g, v in grad_and_vars:
                indices.append(g.indices)
                values.append(g.values)
            all_indices = tf.concat(indices, 0)
            avg_values = tf.concat(values, 0) / len(grad_and_vars)
            # deduplicate across indices
            avg_value, unique_indexs = _deduplicate_indexed_slices(avg_values, all_indices)
            grad = tf.IndexedSlices(values=avg_value, indices=unique_indexs, dense_shape=g0.dense_shape)

        else:
            # a normal tensor can just do a simple average
            grads = []
            for g, v in grad_and_vars:
                # Add 0 dimension to the gradients to represent the tower.
                expanded_g = tf.expand_dims(g, axis=0)
                # Append on a 'tower' dimension which we will average over 
                grads.append(expanded_g)

            # Average over the 'tower' dimension.
            grad = tf.concat(values=grads, axis=0)
            grad = tf.reduce_mean(input_tensor=grad, axis=0)

        # the Variables are redundant because they are shared
        # across towers. So.. just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)

        average_grads.append(grad_and_var)

    assert len(average_grads) == len(list(zip(*tower_grads)))
    return average_grads

def summary_gradient_updates(grads, opt, lr):
    '''get summary ops for the magnitude of gradient updates'''

    # strategy:
    # make a dict of variable name -> [variable, grad, adagrad slot]
    vars_grads = {}
    for v in tf.trainable_variables():
        vars_grads[v.name] = [v, None, None]
    for g, v in grads:
        vars_grads[v.name][1] = g
        vars_grads[v.name][2] = opt.get_slot(v, 'accumulator')

    # now make summaries
    ret = []
    for vname, (v, g, accumulator) in vars_grads.items():

        if g is None:
            continue

        if isinstance(g, tf.IndexedSlices):
            # a sparse gradient - only take norm of params that are updated
            values = tf.gather(v, g.indices)
            updates = lr * g.values
            if accumulator is not None:
                updates /= tf.sqrt(tf.gather(accumulator, g.indices))
        else:
            values = v
            updates = lr * g
            if accumulator is not None:
                updates /= tf.sqrt(accumulator)

        values_norm = tf.sqrt(tf.reduce_sum(v * v)) + 1.0e-7
        updates_norm = tf.sqrt(tf.reduce_sum(updates * updates))
        ret.append(tf.summary.scalar('UPDATE/' + vname.replace(":", "_"), updates_norm / values_norm))

    return ret

def _deduplicate_indexed_slices(values, indices):
    """Sums `values` associated with any non-unique `indices`.
    Args:
      values: A `Tensor` with rank >= 1.
      indices: A one-dimensional integer `Tensor`, indexing into the first
      dimension of `values` (as in an IndexedSlices object).
    Returns:
      A tuple of (`summed_values`, `unique_indices`) where `unique_indices` is a
      de-duplicated version of `indices` and `summed_values` contains the sum of
      `values` slices associated with each unique index.
    """
    unique_indices, new_index_positions = tf.unique(indices)
    summed_values = tf.unsorted_segment_sum(
      data=values,
      segment_ids=new_index_positions,
      num_segments=tf.shape(unique_indices)[0])
    return (summed_values, unique_indices)

def _get_feed_dict_from_X(X, start, end, model, char_inputs, bidirectional):
    feed_dict = {}
    if char_inputs:
        # character inputs
        char_ids = X['tokens_characters'][start:end]
        feed_dict[model.tokens_characters] = char_ids
    else:
        token_ids = X['token_ids'][start:end]
        feed_dict[model.token_ids] = token_ids

    if bidirectional:
        if char_inputs:
            feed_dict[model.tokens_characters_reverse] = \
                X['tokens_characters_reverse'][start:end]
        else:
            feed_dict[model.token_ids_reverse] = \
                X['token_ids_reverse'][start:end]

    # now the targets with weights
    next_id_placeholders = [[model.next_token_id, '']]
    if bidirectional:
        next_id_placeholders.append([model.next_token_id_reverse, '_reverse'])

    for id_placeholder, suffix in next_id_placeholders:
        name = 'next_token_id' + suffix
        feed_dict[id_placeholder] = X[name][start:end]
    return feed_dict


def train(options, data, n_gpus, tf_save_dir, tf_log_dir,
          restart_ckpt_file=None):

    # not restarting so save the options
    if restart_ckpt_file is None:
        with open(os.path.join(tf_save_dir, 'options.json'), 'w') as fout:
            fout.write(json.dumps(options))

    with tf.device('/cpu:0'):
        global_step = tf.get_variable(
            'global_step', [],
            initializer=tf.constant_initializer(0), trainable=False)

        # set up the optimizer
        lr = options.get('learning_rate', 0.2)
        optimizer = tf.train.AdagradOptimizer(learning_rate=lr,
                                              initial_accumulator_value=1.0)

        # calculate the gradients on each GPU
        tower_grad_vars = []
        model_of_gpus = []
        train_perplexity = tf.get_variable(
            'train_perplexity', [],
            initializer=tf.constant_initializer(0.0), trainable=False)
        norm_summaries = []
        for k in range(n_gpus):
            with tf.device('/gpu:%d' % k): # 每个gpu建立一个语言模型,然后参数共享
                with tf.variable_scope('lm', reuse=k > 0):  # 妙,太妙了
                    # calculate the loss for one model replica and get
                    #   lstm states
                    model = LanguageModel(options, is_training=True)
                    loss = model.total_loss
                    model_of_gpus.append(model)
                    # get gradients
                    grad_vars = optimizer.compute_gradients(
                        loss * options['unroll_steps'], # unroll_steps:20
                        aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE,
                    )
                    tower_grad_vars.append(grad_vars)
                    # keep track of loss across all GPUs
                    train_perplexity += loss
        # 打印所有变量与成员
        print_variable_summary()

        # calculate the mean of each gradient across all GPUs
        print("tower_grad_vars:", tower_grad_vars)
        # tower_grad_vars:
        # [
        # [(grad0_gpu0, var0_gpu0), (grad1_gpu0, var1_gpu0),.., (gradN_gpu1, varN_gpu1) ],
        # [(grad0_gpu1, var0_gpu1), (grad1_gpu1, var1_gpu1),.., (gradN_gpu2, varN_gpu2) ],
        # ...
        # [(grad0_gpuk, var0_gpuk), (grad1_gpuk, var1_gpuk),.., (gradN_gpuk, varN_gpuk) ],
        # ]
        grad_vars = average_gradients(tower_grad_vars, options['batch_size'], options)
        grad_vars, norm_summary_ops = clip_grads(grad_vars, options, do_summaries=True, global_step=global_step)
        norm_summaries.extend(norm_summary_ops)

        # log the training perplexity
        train_perplexity = tf.exp(train_perplexity / n_gpus) # 困惑度
        perplexity_summmary = tf.summary.scalar('train_perplexity', train_perplexity)

        # some histogram summaries.  all models use the same parameters
        # so only need to summarize one
        histogram_summaries = [
            tf.summary.histogram('token_embedding', model_of_gpus[0].embedding)
        ]
        # tensors of the output from the LSTM layer
        lstm_out = tf.get_collection('lstm_output_embeddings')
        histogram_summaries.append(tf.summary.histogram('lstm_embedding_0', lstm_out[0]))
        if options.get('bidirectional', False):
            # also have the backward embedding
            histogram_summaries.append(tf.summary.histogram('lstm_embedding_1', lstm_out[1]))

        # apply the gradients to create the training operation
        train_op = optimizer.apply_gradients(grad_vars, global_step=global_step)

        # histograms of variables
        for v in tf.global_variables():
            histogram_summaries.append(tf.summary.histogram(v.name.replace(":", "_"), v))

        # get the gradient updates -- these aren't histograms, but we'll
        # only update them when histograms are computed
        histogram_summaries.extend(summary_gradient_updates(grad_vars, optimizer, lr))

        saver = tf.train.Saver(tf.global_variables(), max_to_keep=2)
        summary_op = tf.summary.merge(
            [perplexity_summmary] + norm_summaries
        )
        hist_summary_op = tf.summary.merge(histogram_summaries)

        init = tf.initialize_all_variables()

    # do the training loop
    is_bidirectional = options.get('bidirectional', False)
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        sess.run(init)

        # load the checkpoint data if needed
        if restart_ckpt_file is not None:
            saver = tf.train.Saver()
            saver.restore(sess, restart_ckpt_file)
            
        summary_writer = tf.summary.FileWriter(tf_log_dir, sess.graph)

        # For each batch:
        # Get a batch of data from the generator. The generator will
        # yield batches of size batch_size * n_gpus that are sliced
        # and fed for each required placeholer.
        #
        # We also need to be careful with the LSTM states.  We will
        # collect the final LSTM states after each batch, then feed
        # them back in as the initial state for the next batch

        batch_size = options['batch_size']
        unroll_steps = options['unroll_steps']
        n_train_tokens = options.get('n_train_tokens', 768648884)
        n_tokens_per_batch = batch_size * unroll_steps * n_gpus # 每个batch生成的数据有n_gpus份
        n_batches_per_epoch = int(n_train_tokens / n_tokens_per_batch)
        n_batches_total = options['n_epochs'] * n_batches_per_epoch
        print("Training for %s epochs and %s batches" % (
            options['n_epochs'], n_batches_total))

        # get the initial lstm states
        init_state_tensors = []
        final_state_tensors = []
        for model in model_of_gpus:
            init_state_tensors.extend(model.init_lstm_state)
            final_state_tensors.extend(model.final_lstm_state)

        char_inputs = 'char_cnn' in options
        if char_inputs:
            max_char_count_per_token = options['char_cnn']['max_characters_per_token']
            feed_dict = {
                model.tokens_characters:
                    np.zeros([batch_size, unroll_steps, max_char_count_per_token], dtype=np.int32)
                for model in model_of_gpus
            }
        else:
            feed_dict = {
                model.token_ids:
                    np.zeros([batch_size, unroll_steps], dtype=np.int64)
                for model in model_of_gpus
            }

        if is_bidirectional:
            if char_inputs:
                feed_dict.update({
                    model.tokens_characters_reverse:
                        np.zeros([batch_size, unroll_steps, max_char_count_per_token],
                                 dtype=np.int32)
                    for model in model_of_gpus
                })
            else:
                feed_dict.update({
                    model.token_ids_reverse:
                        np.zeros([batch_size, unroll_steps], dtype=np.int64)
                    for model in model_of_gpus
                })
        # lstm
        init_state_values = sess.run(init_state_tensors, feed_dict=feed_dict)
        print("begin to train...")
        t1 = time.time()
        # data_gen:{
        #   token_ids = (batch_size*n_gpus, num_steps)
        #   char_inputs = (batch_size*n_gpus, num_steps, max_char_count_per_token=50)
        #   targets = word ID of next word (batch_size*n_gpus, num_steps)
        # }
        data_gen = data.iter_batches(batch_size * n_gpus, unroll_steps) # 由于是多个gpu,所以需要获取多份数据
        for batch_no, batch in enumerate(data_gen, start=1):
            # slice the input in the batch for the feed_dict
            # 第一次将0作为 init_state_values -> init_state_tensors
            feed_dict = {t: v for t, v in zip(init_state_tensors, init_state_values)}
            X = batch
            for k in range(n_gpus):
                model = model_of_gpus[k]
                start = k * batch_size
                end = (k + 1) * batch_size

                feed_dict.update(
                    _get_feed_dict_from_X(X, start, end, model,
                                          char_inputs, is_bidirectional)
                )

            # This runs the train_op, summaries and the "final_state_tensors"
            #   which just returns the tensors, passing in the initial
            #   state tensors, token ids and next token ids
            if batch_no % 1250 != 0:
                ret = sess.run(
                    [train_op, summary_op, train_perplexity] +
                                                final_state_tensors, # final_state_tensors:list
                    feed_dict=feed_dict
                )

                # first three entries of ret are:
                #  train_op, summary_op, train_perplexity
                # last entries are the final states -- set them to
                # init_state_values for next batch
                """ 
                每一个batch中,会将上一次的hidden_state作为本次lstm的初始化状态 ,
                但个人理解,只有当句子比较长,被多次截断的情况下,前后两个batch之间才有关联,
                而当前后两个batch之间没有关联时, 意义就不大 
                """
                init_state_values = ret[3:]
            else:
                # also run the histogram summaries
                ret = sess.run(
                    [train_op, summary_op, train_perplexity, hist_summary_op] +
                                                final_state_tensors,
                    feed_dict=feed_dict
                )
                # hist_summary
                summary_writer.add_summary(ret[3], batch_no)
                init_state_values = ret[4:]

            if batch_no % 20 == 0:
                # ret:[train_op, summary_op, train_perplexity]
                # write the summaries to tensorboard and display perplexity
                summary_writer.add_summary(ret[1], batch_no)
                print("Batch %s, train_perplexity=%s" % (batch_no, ret[2]))
                print("Total time: %s" % (time.time() - t1))

            if (batch_no % 1250 == 0) or (batch_no == n_batches_total):
                # save the model
                checkpoint_path = os.path.join(tf_save_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=global_step)

            if batch_no == n_batches_total: # 这里有epoch的数量
                # done training!
                print("done training!")
                break

def clip_by_global_norm_summary(t_list, clip_norm, norm_name, variables):
    # wrapper around tf.clip_by_global_norm that also does summary ops of norms

    # compute norms
    # use global_norm with one element to handle IndexedSlices vs dense
    norms = [tf.global_norm([t]) for t in t_list]

    # summary ops before clipping
    summary_ops = []
    for ns, v in zip(norms, variables):
        name = 'norm_pre_clip/' + v.name.replace(":", "_")
        summary_ops.append(tf.summary.scalar(name, ns))

    # clip 
    clipped_t_list, tf_norm = tf.clip_by_global_norm(t_list, clip_norm)

    # summary ops after clipping
    norms_post = [tf.global_norm([t]) for t in clipped_t_list]
    for ns, v in zip(norms_post, variables):
        name = 'norm_post_clip/' + v.name.replace(":", "_")
        summary_ops.append(tf.summary.scalar(name, ns))

    summary_ops.append(tf.summary.scalar(norm_name, tf_norm))

    return clipped_t_list, tf_norm, summary_ops


def clip_grads(grads_and_vars_list, options, do_summaries, global_step):
    # grads = [(grad1, var1), (grad2, var2), ...]
    def _clip_norms(grad_and_vars, val, name):
        # grad_and_vars is a list of (g, v) pairs
        grad_tensors = [g for g, v in grad_and_vars]
        v_list = [v for g, v in grad_and_vars]
        scaled_val = val
        if do_summaries:
            clipped_tensors, g_norm, summary_ops = clip_by_global_norm_summary(grad_tensors, scaled_val, name, v_list)
        else:
            summary_ops = []
            # clip_by_global_norm:需要等所有tensor的梯度计算完毕,才能计算,速度会慢一些
            clipped_tensors, g_norm = tf.clip_by_global_norm(t_list=grad_tensors, clip_norm=scaled_val)

        clipped_and_original_tensor = []
        for clipped_tensor, (g, v) in zip(clipped_tensors, grad_and_vars):
            clipped_and_original_tensor.append((clipped_tensor, v))

        return clipped_and_original_tensor, summary_ops

    all_clip_norm_val = options['all_clip_norm_val']
    ret, summary_ops = _clip_norms(grads_and_vars_list, all_clip_norm_val, 'norm_grad')

    assert len(ret) == len(grads_and_vars_list)

    return ret, summary_ops


def test(options, ckpt_file, data, batch_size=256):
    '''
    Get the test set perplexity!
    '''
    print("get test set perplexity!")
    bidirectional = options.get('bidirectional', False)
    char_inputs = 'char_cnn' in options
    if char_inputs:
        max_chars = options['char_cnn']['max_characters_per_token']

    unroll_steps = 1

    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        with tf.device('/gpu:0'), tf.variable_scope('lm'):
            test_options = dict(options)
            # NOTE: the number of tokens we skip in the last incomplete
            # batch is bounded above batch_size * unroll_steps
            test_options['batch_size'] = batch_size
            test_options['unroll_steps'] = 1
            model = LanguageModel(test_options, is_training=False)
            # we use the "Saver" class to load the variables
            loader = tf.train.Saver()
            loader.restore(sess, ckpt_file)

        # model.total_loss is the op to compute the loss
        # perplexity is exp(loss)
        init_state_tensors = model.init_lstm_state
        final_state_tensors = model.final_lstm_state
        if char_inputs:
            feed_dict = {
                model.token_ids:
                        np.zeros([batch_size, unroll_steps], dtype=np.int64)
            }
            if bidirectional:
                feed_dict.update({
                    model.token_ids_reverse:
                        np.zeros([batch_size, unroll_steps], dtype=np.int64)
                })  
        else:
            feed_dict = {
                model.tokens_characters:
                   np.zeros([batch_size, unroll_steps, max_chars],
                                 dtype=np.int32)
            }
            if bidirectional:
                feed_dict.update({
                    model.tokens_characters_reverse:
                        np.zeros([batch_size, unroll_steps, max_chars],
                            dtype=np.int32)
                })

        init_state_values = sess.run(
            init_state_tensors,
            feed_dict=feed_dict)

        t1 = time.time()
        batch_losses = []
        total_loss = 0.0
        for batch_no, batch in enumerate(
                                data.iter_batches(batch_size, 1), start=1):
            # slice the input in the batch for the feed_dict
            X = batch

            feed_dict = {t: v for t, v in zip(
                                        init_state_tensors, init_state_values)}

            feed_dict.update(
                _get_feed_dict_from_X(X,
                                      start=0,
                                      end=X['token_ids'].shape[0],
                                      model=model,
                                      char_inputs=char_inputs,
                                      bidirectional=bidirectional)
            )

            ret = sess.run(
                [model.total_loss, final_state_tensors],
                feed_dict=feed_dict
            )

            loss, init_state_values = ret
            batch_losses.append(loss)
            batch_perplexity = np.exp(loss) # perplexity, exp(cross-entropy)
            total_loss += loss
            avg_perplexity = np.exp(total_loss / batch_no)

            print("batch=%s, batch_perplexity=%s, avg_perplexity=%s, time=%s" %
                (batch_no, batch_perplexity, avg_perplexity, time.time() - t1))

    avg_loss = np.mean(batch_losses)
    print("FINSIHED!  AVERAGE PERPLEXITY = %s" % np.exp(avg_loss))

    return np.exp(avg_loss)


def load_options_latest_checkpoint(tf_save_dir):
    options_file = os.path.join(tf_save_dir, 'options.json')
    with open(options_file, 'r') as fin:
        options = json.load(fin)

    ckpt_file = tf.train.latest_checkpoint(tf_save_dir)
    return options, ckpt_file


def load_vocab(vocab_file, max_char_count_in_token=None):
    if max_char_count_in_token:
        return UnicodeCharsVocabulary(vocab_file, max_char_count_in_token,
                                      validate_file=True)
    else:
        return Vocabulary(vocab_file, validate_file=True)


def dump_weights(tf_save_dir, outfile):
    '''
    Dump the trained weights from a model to a HDF5 file.
    '''
    import h5py

    # 为何要将名称替换?
    def _get_outname(tf_name):
        outname = re.sub(':0$', '', tf_name)
        outname = outname.lstrip('lm/')
        outname = re.sub('/rnn/', '/RNN/', outname)
        outname = re.sub('/multi_rnn_cell/', '/MultiRNNCell/', outname)
        outname = re.sub('/cell_', '/Cell', outname)
        outname = re.sub('/lstm_cell/', '/LSTMCell/', outname)
        if '/RNN/' in outname:
            if 'projection' in outname:
                outname = re.sub('projection/kernel', 'W_P_0', outname)
            else:
                outname = re.sub('/kernel', '/W_0', outname)
                outname = re.sub('/bias', '/B', outname)
        return outname

    options, ckpt_file = load_options_latest_checkpoint(tf_save_dir)

    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        with tf.variable_scope('lm'):
            # 先建立model graph
            model = LanguageModel(options, False)
            # we use the "Saver" class to load the variables
            saver = tf.train.Saver()
            saver.restore(sess, ckpt_file)

        with h5py.File(outfile, 'w') as fout:
            for var in tf.trainable_variables():
                if var.name.find('softmax') >= 0:
                    # don't dump these
                    continue
                outname = _get_outname(var.name)
                print("Saving variable {0} with name {1}".format(var.name, outname))
                shape = var.get_shape().as_list()
                dset = fout.create_dataset(outname, shape, dtype='float32')
                values = sess.run([var])[0]
                dset[...] = values
