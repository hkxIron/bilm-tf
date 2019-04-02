import numpy as np
import tensorflow as tf
import h5py
import json
import re

from .data import UnicodeCharsVocabulary, CharBatcher

DTYPE = 'float32'
DTYPE_INT = 'int64'


class BidirectionalLanguageModel(object):
    def __init__(
            self,
            options_file: str,
            weight_file: str,
            use_character_inputs=True,
            embedding_weight_file=None,
            max_batch_size=128,
        ):
        '''
        Creates the language model computational graph and loads weights

        Two options for input type:
            (1) To use character inputs (paired with Batcher)
                pass use_character_inputs=True, and ids_placeholder
                of shape (None, None, max_characters_per_token)
                to __call__
            (2) To use token ids as input (paired with TokenBatcher),
                pass use_character_inputs=False and ids_placeholder
                of shape (None, None) to __call__.
                In this case, embedding_weight_file is also required input

        options_file: location of the json formatted file with
                      LM hyperparameters
        weight_file: location of the hdf5 file with LM weights
        use_character_inputs: if True, then use character ids as input,
            otherwise use token ids
        max_batch_size: the maximum allowable batch size 
        '''
        with open(options_file, 'r') as fin:
            options = json.load(fin)

        # 如果不使用char_embedding,需要有token_embedding的file
        if not use_character_inputs:
            if embedding_weight_file is None:
                raise ValueError(
                    "embedding_weight_file is required input with "
                    "not use_character_inputs"
                )

        self._options = options
        self._weight_file = weight_file
        self._embedding_weight_file = embedding_weight_file
        self._use_character_inputs = use_character_inputs
        self._max_batch_size = max_batch_size

        self._ops = {}
        self._graphs = {}

    def __call__(self, ids_placeholder):
        '''
        Given the input character ids (or token ids), returns a dictionary
            with tensorflow ops:

            {'lm_embeddings': embedding_op,
             'lengths': sequence_lengths_op,
             'mask': op to compute mask}

        embedding_op computes the LM embeddings and is shape
            (None, 3, None, 1024) => [batch, layer_num, time_steps, embedding_size]
        lengths_op computes the sequence lengths and is shape (None, ) => [batch]
        mask computes the sequence mask and is shape (None, None), => 即[batch, time_steps]

        ids_placeholder: a tf.placeholder of type int32.
            If use_character_inputs=True, it is shape
                (None, None, max_characters_per_token) and holds the input
                character ids for a batch
            If use_character_input=False, it is shape (None, None) and
                holds the input token ids for a batch
        '''
        if ids_placeholder in self._ops:
            # have already created ops for this placeholder, just return them
            ret = self._ops[ids_placeholder]

        else:
            # need to create the graph
            if len(self._ops) == 0:
                # first time creating the graph, don't reuse variables
                lm_graph = BidirectionalLanguageModelGraph(
                    self._options,
                    self._weight_file,
                    ids_placeholder,
                    embedding_weight_file=self._embedding_weight_file,
                    use_character_inputs=self._use_character_inputs,
                    max_batch_size=self._max_batch_size)
            else:
                # reuse variables
                with tf.variable_scope('', reuse=True):
                    lm_graph = BidirectionalLanguageModelGraph(
                        self._options,
                        self._weight_file,
                        ids_placeholder,
                        embedding_weight_file=self._embedding_weight_file,
                        use_character_inputs=self._use_character_inputs,
                        max_batch_size=self._max_batch_size)

            ops = self._build_ops(lm_graph)
            self._ops[ids_placeholder] = ops
            self._graphs[ids_placeholder] = lm_graph
            ret = ops

        return ret

    def _build_ops(self, lm_graph):
        with tf.control_dependencies([lm_graph.update_state_op]):
            # get the LM embeddings
            # embedding:[batch, unroll_steps, projection_dim]
            token_embeddings = lm_graph.embedding
            # layers:[ [batch, unroll_steps, projection_dim*2] ]
            # 1.先将原始的embedding层加入,一个是前向,一个是后向
            layers = [
                tf.concat([token_embeddings, token_embeddings], axis=2)
            ]

            n_lm_layers = len(lm_graph.lstm_outputs['forward'])
            # 2.加入不同层的 embedding
            # lstm不同的层数,不同层代表不同的特层
            # 一般认为低层代表词法信息,高层代表语义信息
            for i in range(n_lm_layers):
                layers.append(
                    tf.concat(
                        [lm_graph.lstm_outputs['forward'][i],
                         lm_graph.lstm_outputs['backward'][i]],
                        axis=-1)
                )

            # The layers include the BOS/EOS tokens.  Remove them
            sequence_length_wo_bos_eos = lm_graph.sequence_lengths - 2
            layers_without_bos_eos = []
            for layer in layers:
                # 去掉首尾的bos/eos居然用这种奇淫巧技
                # layer:[batch, unroll_steps, projection_dim*2]
                layer_wo_bos_eos = layer[:, 1:, :]
                layer_wo_bos_eos = tf.reverse_sequence(
                    layer_wo_bos_eos, 
                    lm_graph.sequence_lengths - 1,
                    seq_axis=1,
                    batch_axis=0,
                )
                layer_wo_bos_eos = layer_wo_bos_eos[:, 1:, :]
                layer_wo_bos_eos = tf.reverse_sequence(
                    layer_wo_bos_eos,
                    sequence_length_wo_bos_eos,
                    seq_axis=1,
                    batch_axis=0,
                )
                layers_without_bos_eos.append(layer_wo_bos_eos)

            # concatenate the layers
            # t:[batch, unroll_steps, projection_dim*2]
            # lm_embeddings:[batch, layer_num, unroll_steps, projection_dim*2]
            lm_embeddings = tf.concat(
                [tf.expand_dims(t, axis=1) for t in layers_without_bos_eos],
                axis=1
            )

            # get the mask op without bos/eos.
            # tf doesn't support reversing boolean tensors, so cast
            # to int then back
            # 奇淫巧技
            mask_wo_bos_eos = tf.cast(lm_graph.mask[:, 1:], 'int32')
            mask_wo_bos_eos = tf.reverse_sequence(
                mask_wo_bos_eos,
                lm_graph.sequence_lengths - 1,
                seq_axis=1,
                batch_axis=0,
            )
            mask_wo_bos_eos = mask_wo_bos_eos[:, 1:]
            mask_wo_bos_eos = tf.reverse_sequence(
                mask_wo_bos_eos,
                sequence_length_wo_bos_eos,
                seq_axis=1,
                batch_axis=0,
            )
            mask_wo_bos_eos = tf.cast(mask_wo_bos_eos, 'bool')

        return {
            'lm_embeddings': lm_embeddings, # lm_embeddings:[batch, layer_num, unroll_steps, projection_dim*2]
            'lengths': sequence_length_wo_bos_eos,
            'token_embeddings': lm_graph.embedding, # embedding:[batch, unroll_steps, projection_dim]
            'mask': mask_wo_bos_eos, # mask:[batch, unroll_step-2]
        }


def _pretrained_initializer(varname:str, weight_file:str, embedding_weight_file=None):
    '''
    We'll stub out(去掉) all the initializers in the pretrained LM with
    a function that loads the weights from the file

    我们将去掉模型中的所有初始值设定项。
    '''

    """
     ['lm/RNN_0/rnn/multi_rnn_cell/cell_0/lstm_cell/bias:0',  TensorShape([Dimension(120)])],
     ['lm/RNN_0/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel:0',  TensorShape([Dimension(20),Dimension(120)])],
     ['lm/RNN_0/rnn/multi_rnn_cell/cell_0/lstm_cell/projection/kernel:0',  TensorShape([Dimension(30),Dimension(10)])],

     ['lm/RNN_0/rnn/multi_rnn_cell/cell_1/lstm_cell/bias:0',  TensorShape([Dimension(120)])],
     ['lm/RNN_0/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel:0',  TensorShape([Dimension(20),Dimension(120)])],
     ['lm/RNN_0/rnn/multi_rnn_cell/cell_1/lstm_cell/projection/kernel:0',  TensorShape([Dimension(30),Dimension(10)])],

     ['lm/RNN_1/rnn/multi_rnn_cell/cell_0/lstm_cell/bias:0',  TensorShape([Dimension(120)])],
     ['lm/RNN_1/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel:0',  TensorShape([Dimension(20),Dimension(120)])],
     ['lm/RNN_1/rnn/multi_rnn_cell/cell_0/lstm_cell/projection/kernel:0',  TensorShape([Dimension(30),Dimension(10)])],

     ['lm/RNN_1/rnn/multi_rnn_cell/cell_1/lstm_cell/bias:0',  TensorShape([Dimension(120)])],
     ['lm/RNN_1/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel:0',  TensorShape([Dimension(20),Dimension(120)])],
     ['lm/RNN_1/rnn/multi_rnn_cell/cell_1/lstm_cell/projection/kernel:0',  TensorShape([Dimension(30),Dimension(10)])],
    """

    weight_name_map = {}
    for i in range(2): # i为正向或者反向
        for j in range(8):  # if we decide to add more layers, lstm stacked的层数
            root = 'RNN_{}/RNN/MultiRNNCell/Cell{}'.format(i, j)
            weight_name_map[root + '/rnn/lstm_cell/kernel'] =  root + '/LSTMCell/W_0'
            weight_name_map[root + '/rnn/lstm_cell/bias'] = root + '/LSTMCell/B'
            weight_name_map[root + '/rnn/lstm_cell/projection/kernel'] = root + '/LSTMCell/W_P_0'

    # convert the graph name to that in the checkpoint
    varname_in_file = varname[5:] # str
    if varname_in_file.startswith('RNN'):
        varname_in_file = weight_name_map[varname_in_file]

    if varname_in_file == 'embedding':
        with h5py.File(embedding_weight_file, 'r') as fin:
            # Have added a special 0 index for padding not present
            # in the original model.
            embed_weights = fin[varname_in_file][...]
            weights = np.zeros((embed_weights.shape[0] + 1, embed_weights.shape[1]),
                dtype=DTYPE
            )
            weights[1:, :] = embed_weights
    else:
        with h5py.File(weight_file, 'r') as fin:
            if varname_in_file == 'char_embed':
                # Have added a special 0 index for padding not present
                # in the original model.
                char_embed_weights = fin[varname_in_file][...]
                weights = np.zeros(
                    (char_embed_weights.shape[0] + 1, char_embed_weights.shape[1]),
                    dtype=DTYPE
                )
                weights[1:, :] = char_embed_weights
            else:
                weights = fin[varname_in_file][...]

    # Tensorflow initializers are callables that accept a shape parameter
    # and some optional kwargs
    def ret(shape, **kwargs):
        # 用这个函数,就是为了检查shape是否对得上
        if list(shape) != list(weights.shape):
            raise ValueError(
                "Invalid shape initializing {0}, got {1}, expected {2}".format(
                    varname_in_file, shape, weights.shape)
            )
        return weights

    return ret


class BidirectionalLanguageModelGraph(object):
    '''
    Creates the computational graph and holds the ops necessary for runnint
    a bidirectional language model
    '''
    def __init__(self, options, weight_file, ids_placeholder,
                 use_character_inputs=True,
                 embedding_weight_file=None,
                 max_batch_size=128):

        self.options = options
        self._max_batch_size = max_batch_size
        self.ids_placeholder = ids_placeholder
        self.use_character_inputs = use_character_inputs

        # this custom_getter will make all variables not trainable and
        # override the default initializer
        def custom_getter(getter, name:str, *args, **kwargs):
            kwargs['trainable'] = False
            kwargs['initializer'] = _pretrained_initializer(name, weight_file, embedding_weight_file)
            return getter(name, *args, **kwargs)

        if embedding_weight_file:
            # get the vocab size
            with h5py.File(embedding_weight_file, 'r') as fin:
                # +1 for padding
                self._n_tokens_vocab = fin['embedding'].shape[0] + 1
        else:
            self._n_tokens_vocab = None

        with tf.variable_scope('bilm', custom_getter=custom_getter):
            self._build()

    def _build(self):
        if self.use_character_inputs:
            self._build_word_char_embeddings()
        else:
            self._build_word_embeddings()
        self._build_lstms()

    # 此函数与training.py中类似,可以参考它的注释
    def _build_word_char_embeddings(self):
        '''
        options contains key 'char_cnn': {

        'n_characters': 262,

        # includes the start / end characters
        'max_characters_per_token': 50,

        'filters': [
            [1, 32],  # kernel=1,filter_count=32
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
        projection_dim = self.options['lstm']['projection_dim']

        cnn_options = self.options['char_cnn']
        filters = cnn_options['filters']
        n_filters = sum(f[1] for f in filters) # filter:[kernel,num]
        max_chars = cnn_options['max_characters_per_token']
        char_embed_dim = cnn_options['embedding']['dim']
        n_chars = cnn_options['n_characters']
        if n_chars != 262:
            raise InvalidNumberOfCharacters(
                "Set n_characters=262 after training see the README.md"
            )
        if cnn_options['activation'] == 'tanh':
            activation = tf.nn.tanh
        elif cnn_options['activation'] == 'relu':
            activation = tf.nn.relu

        # the character embeddings
        with tf.device("/cpu:0"):
            self.embedding_weights = tf.get_variable(
                    "char_embed", [n_chars, char_embed_dim],
                    dtype=DTYPE,
                    initializer=tf.random_uniform_initializer(-1.0, 1.0)
            )
            # shape (batch_size, unroll_steps, max_chars, embed_dim)
            self.char_embedding = tf.nn.embedding_lookup(self.embedding_weights,
                                                    self.ids_placeholder)

        # 与training.py不同的时,这里不需要bi-lstm了

        # the convolutions
        def make_convolutions(input):
            with tf.variable_scope('CNN') as scope:
                convolutions = []
                for i, (width, num) in enumerate(filters):
                    # filters:[ [1, 32], ... ] # kernel=1,filter_count=32
                    if cnn_options['activation'] == 'relu':
                        # He initialization for ReLU activation
                        # with char embeddings init between -1 and 1
                        #w_init = tf.random_normal_initializer(
                        #    mean=0.0,
                        #    stddev=np.sqrt(2.0 / (width * char_embed_dim))
                        #)

                        # Kim et al 2015, +/- 0.05
                        w_init = tf.random_uniform_initializer( minval=-0.05, maxval=0.05)
                    elif cnn_options['activation'] == 'tanh':
                        # glorot init
                        w_init = tf.random_normal_initializer(
                            mean=0.0,
                            stddev=np.sqrt(1.0 / (width * char_embed_dim))
                        )
                    w = tf.get_variable(
                        "W_cnn_%s" % i,
                        [1, width, char_embed_dim, num],
                        initializer=w_init,
                        dtype=DTYPE)
                    b = tf.get_variable(
                        "b_cnn_%s" % i, [num], dtype=DTYPE,
                        initializer=tf.constant_initializer(0.0))

                    conv = tf.nn.conv2d(
                            input, w,
                            strides=[1, 1, 1, 1],
                            padding="VALID") + b
                    # now max pool
                    conv = tf.nn.max_pool(
                            conv, [1, 1, max_chars-width+1, 1],
                            [1, 1, 1, 1], 'VALID')

                    # activation
                    conv = activation(conv)
                    conv = tf.squeeze(conv, squeeze_dims=[2])

                    # convolutions:[ [batch, unroll_steps, filter_num], [b, u, f], ... ]
                    convolutions.append(conv)

            # convolutions:[batch, unroll_steps, total_filter_num]
            return tf.concat(convolutions, 2)

        # char_embedding: [batch_size, unroll_steps, max_chars, embed_dim]
        # embedding:[batch, unroll_steps, total_filter_num]
        embedding = make_convolutions(self.char_embedding)

        # for highway and projection layers
        n_highway = cnn_options.get('n_highway')
        use_highway = n_highway is not None and n_highway > 0
        use_proj = n_filters != projection_dim

        if use_highway or use_proj:
            #   reshape from (batch_size, token_count_per_sentence, dim) to (-1, dim)
            batch_size_n_tokens = tf.shape(embedding)[0:2]
            # embedding:[batch*unroll_steps=token_count, total_filter_num]
            embedding = tf.reshape(embedding, [-1, n_filters])

        # set up weights for projection
        if use_proj:
            assert n_filters > projection_dim
            with tf.variable_scope('CNN_proj') as scope:
                    W_proj_cnn = tf.get_variable(
                        "W_proj", [n_filters, projection_dim],
                        initializer=tf.random_normal_initializer(
                            mean=0.0, stddev=np.sqrt(1.0 / n_filters)),
                        dtype=DTYPE)
                    b_proj_cnn = tf.get_variable(
                        "b_proj", [projection_dim],
                        initializer=tf.constant_initializer(0.0),
                        dtype=DTYPE)

        # apply highways layers
        def highway(x, ww_carry, bb_carry, ww_tr, bb_tr):
            carry_gate = tf.nn.sigmoid(tf.matmul(x, ww_carry) + bb_carry)
            transform_gate = tf.nn.relu(tf.matmul(x, ww_tr) + bb_tr)
            return carry_gate * transform_gate + (1.0 - carry_gate) * x

        if use_highway:
            highway_dim = n_filters

            for i in range(n_highway):
                with tf.variable_scope('CNN_highway_%s' % i) as scope:
                    W_carry = tf.get_variable(
                        'W_carry', [highway_dim, highway_dim],
                        # glorit init
                        initializer=tf.random_normal_initializer(
                            mean=0.0, stddev=np.sqrt(1.0 / highway_dim)),
                        dtype=DTYPE)
                    b_carry = tf.get_variable(
                        'b_carry', [highway_dim],
                        initializer=tf.constant_initializer(-2.0),
                        dtype=DTYPE)
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

        # finally project down if needed
        if use_proj:
            # embedding:[batch*unroll_steps, total_filter_num]
            # => [batch*unroll_steps, projection_dim]
            embedding = tf.matmul(embedding, W_proj_cnn) + b_proj_cnn

        # reshape back to (batch_size, tokens, dim)
        if use_highway or use_proj:
            shp = tf.concat([batch_size_n_tokens, [projection_dim]], axis=0)
            embedding = tf.reshape(embedding, shp)

        # at last assign attributes for remainder of the model
        # embedding:[batch, unroll_steps, projection_dim]
        self.embedding = embedding


    def _build_word_embeddings(self):
        projection_dim = self.options['lstm']['projection_dim']

        # the word embeddings
        with tf.device("/cpu:0"):
            self.embedding_weights = tf.get_variable(
                "embedding", [self._n_tokens_vocab, projection_dim],
                dtype=DTYPE,
            )
            self.embedding = tf.nn.embedding_lookup(self.embedding_weights,
                                                self.ids_placeholder)


    def _build_lstms(self):
        # now the LSTMs
        # these will collect the initial states for the forward
        #   (and reverse LSTMs if we are doing bidirectional)

        # parse the options
        lstm_dim = self.options['lstm']['dim']
        projection_dim = self.options['lstm']['projection_dim']
        n_lstm_layers = self.options['lstm'].get('n_layers', 1)
        cell_clip = self.options['lstm'].get('cell_clip')
        proj_clip = self.options['lstm'].get('proj_clip')
        use_skip_connections = self.options['lstm']['use_skip_connections']

        if use_skip_connections:
            print("USING SKIP CONNECTIONS")
        else:
            print("NOT USING SKIP CONNECTIONS")

        # the sequence lengths from input mask
        # mask:[batch, unroll_step]
        if self.use_character_inputs:
            # ids:[batch, unroll_step, char_count_per_token]
            mask = tf.reduce_any(self.ids_placeholder > 0, axis=2) # true or false
        else:
            mask = self.ids_placeholder > 0

        # mask:[batch, unroll_step]
        # sequence_lengths:[batch]
        sequence_lengths = tf.reduce_sum(tf.cast(mask, tf.int32), axis=1)
        batch_size = tf.shape(sequence_lengths)[0]

        # for each direction, we'll store tensors for each layer
        self.lstm_outputs = {'forward': [], 'backward': []}
        self.lstm_state_sizes = {'forward': [], 'backward': []}
        self.lstm_init_states = {'forward': [], 'backward': []}
        self.lstm_final_states = {'forward': [], 'backward': []}

        update_ops = []
        for direction in ['forward', 'backward']:
            if direction == 'forward':
                layer_input = self.embedding
            else:
                layer_input = tf.reverse_sequence(
                    input=self.embedding,
                    seq_lengths=sequence_lengths,
                    seq_axis=1,
                    batch_axis=0
                )

            for i in range(n_lstm_layers): # 多层lstm
                if projection_dim < lstm_dim: # project_dim:512, lstm_dim:4096
                    # are projecting down output
                    lstm_cell = tf.nn.rnn_cell.LSTMCell(
                        num_units=lstm_dim,
                        num_proj=projection_dim,
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
                        lstm_cell = tf.nn.rnn_cell.ResidualWrapper(lstm_cell)

                # collect the input state, run the dynamic rnn, collect
                # the output
                state_size = lstm_cell.state_size

                # the LSTMs are stateful.  To support multiple batch sizes,
                # we'll allocate size for states up to max_batch_size,
                # then use the first batch_size entries for each batch
                init_states = [
                    tf.Variable(
                        tf.zeros([self._max_batch_size, dim]),
                        trainable=False
                    )
                    for dim in lstm_cell.state_size
                ]
                batch_init_states = [
                    state[:batch_size, :] for state in init_states
                ]

                if direction == 'forward':
                    i_direction = 0
                else:
                    i_direction = 1
                variable_scope_name = 'RNN_{0}/RNN/MultiRNNCell/Cell{1}'.format(i_direction, i)
                with tf.variable_scope(variable_scope_name):
                    layer_output, final_state = tf.nn.dynamic_rnn(
                        lstm_cell,
                        layer_input,
                        sequence_length=sequence_lengths,
                        initial_state=tf.nn.rnn_cell.LSTMStateTuple(
                            *batch_init_states),
                    )

                self.lstm_state_sizes[direction].append(lstm_cell.state_size)
                self.lstm_init_states[direction].append(init_states)
                self.lstm_final_states[direction].append(final_state)
                if direction == 'forward':
                    self.lstm_outputs[direction].append(layer_output)
                else:
                    self.lstm_outputs[direction].append(
                        tf.reverse_sequence(
                            layer_output,
                            sequence_lengths,
                            seq_axis=1,
                            batch_axis=0
                        )
                    )

                with tf.control_dependencies([layer_output]):
                    # update the initial states
                    for i in range(2):
                        new_state = tf.concat(
                            [final_state[i][:batch_size, :],
                             init_states[i][batch_size:, :]], axis=0)
                        state_update_op = tf.assign(init_states[i], new_state)
                        update_ops.append(state_update_op)
    
                layer_input = layer_output

        self.mask = mask
        self.sequence_lengths = sequence_lengths
        self.update_state_op = tf.group(*update_ops)


def dump_token_embeddings(vocab_file, options_file, weight_file, outfile):
    '''
    Given an input vocabulary file, dump all the token embeddings to the
    outfile.  The result can be used as the embedding_weight_file when
    constructing a BidirectionalLanguageModel.
    '''
    with open(options_file, 'r') as fin:
        options = json.load(fin)
    max_word_length = options['char_cnn']['max_characters_per_token']

    vocab = UnicodeCharsVocabulary(vocab_file, max_word_length)
    batcher = CharBatcher(vocab_file, max_word_length)

    # [batch, token_count_per_sentence, char_count_per_token]
    ids_placeholder = tf.placeholder('int32',
                                     shape=(None, None, max_word_length)
    )
    model = BidirectionalLanguageModel(options_file, weight_file)

    # embedding_op:[batch, unroll_steps, projection_dim]
    embedding_op = model(ids_placeholder)['token_embeddings']

    n_tokens = vocab.size
    embed_dim = int(embedding_op.shape[2])

    embeddings = np.zeros((n_tokens, embed_dim), dtype=DTYPE)

    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        for k in range(n_tokens):
            token = vocab.id_to_word(k)
            # char_ids: [batch, max_token_count_in_sentence, max_char_count_in_token]
            # 一个句子中只有一个单词
            char_ids = batcher.batch_sentences(sentences=[[token]])[0, 1, :].reshape(1, 1, -1)
            embeddings[k, :] = sess.run(embedding_op, feed_dict={ids_placeholder: char_ids}
            )

    with h5py.File(outfile, 'w') as fout:
        ds = fout.create_dataset(
            'embedding', embeddings.shape, dtype='float32', data=embeddings
        )

def dump_bilm_embeddings(vocab_file, dataset_file, options_file,
                         weight_file, outfile):
    with open(options_file, 'r') as fin:
        options = json.load(fin)
    max_word_length = options['char_cnn']['max_characters_per_token']

    vocab = UnicodeCharsVocabulary(vocab_file, max_word_length)
    batcher = CharBatcher(vocab_file, max_word_length)

    ids_placeholder = tf.placeholder('int32',
                                     shape=(None, None, max_word_length)
    )
    model = BidirectionalLanguageModel(options_file, weight_file)
    ops = model(ids_placeholder)

    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        sentence_id = 0
        with open(dataset_file, 'r') as fin, h5py.File(outfile, 'w') as fout:
            for line in fin:
                sentence = line.strip().split()
                # char_ids: [batch, max_token_count_in_sentence, max_char_count_in_token]
                char_ids = batcher.batch_sentences(sentences=[sentence])
                embeddings = sess.run(
                    ops['lm_embeddings'], # lm_embeddings:[batch, layer_num, unroll_steps, projection_dim*2]
                    feed_dict={ids_placeholder: char_ids}
                )
                ds = fout.create_dataset(
                    '{}'.format(sentence_id),
                    embeddings.shape[1:], dtype='float32',
                    data=embeddings[0, :, :, :]
                )

                sentence_id += 1
