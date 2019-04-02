import tensorflow as tf

# elmo会将不同层加权平均起来
def weight_layers(name, bilm_ops, l2_coef=None,
                  use_top_only=False, do_layer_norm=False):
    '''
    Weight the layers of a biLM with trainable scalar weights to
    compute ELMo representations.

    For each output layer, this returns two ops.  The first computes
        a layer specific weighted average of the biLM layers, and
        the second the l2 regularizer loss term.
    The regularization terms are also add to tf.GraphKeys.REGULARIZATION_LOSSES 

    Input:
        name = a string prefix used for the trainable variable names
        bilm_ops = the tensorflow ops returned to compute internal
            representations from a biLM.  This is the return value
            from BidirectionalLanguageModel(...)(ids_placeholder)
        l2_coef: the l2 regularization coefficient $\lambda$.
            Pass None or 0.0 for no regularization.
        use_top_only: if True, then only use the top layer.
        do_layer_norm: if True, then apply layer normalization to each biLM
            layer before normalizing

    Output:
        {
            'weighted_op': op to compute weighted average for output,
            'regularization_op': op to compute regularization term
        }
    '''
    # 权重平方和
    def _l2_regularizer(weights):
        # weigths:(layer_num)
        if l2_coef is not None:
            return l2_coef * tf.reduce_sum(tf.square(weights))
        else:
            return 0.0

    # Get ops for computing LM embeddings and mask
    # lm_embeddings:[batch, layer_num, unroll_steps, projection_dim*2]
    lm_embeddings = bilm_ops['lm_embeddings']
    # mask:[batch, unroll_step-2]
    mask = bilm_ops['mask']

    n_lm_layers = int(lm_embeddings.get_shape()[1])
    lm_dim = int(lm_embeddings.get_shape()[3])

    with tf.control_dependencies([lm_embeddings, mask]):
        # Cast the mask and broadcast for layer use.
        mask_float = tf.cast(mask, 'float32')
        # broadcast_mask:[batch, unroll_step-2, 1]
        broadcast_mask = tf.expand_dims(mask_float, axis=-1)

        def _do_layer_normalization(x):
            # do layer normalization excluding the mask
            # broadcast_mask:[batch, unroll_step-2, 1]
            x_masked = x * broadcast_mask
            N = tf.reduce_sum(mask_float) * lm_dim
            mean = tf.reduce_sum(x_masked) / N
            variance = tf.reduce_sum(((x_masked - mean) * broadcast_mask)**2) / N
            return tf.nn.batch_normalization(x, mean, variance, None, None, 1E-12)

        if use_top_only:
            # lm_embeddings:[batch, layer_num, unroll_steps, projection_dim*2]
            # layers:list of [batch, 1, unroll_steps, projection_dim*2]
            layers = tf.split(lm_embeddings, n_lm_layers, axis=1)
            # just the top layer
            # sum_pieces:[batch, 1, unroll_steps, projection_dim*2]
            weighted_layer = tf.squeeze(layers[-1], squeeze_dims=1)
            # no regularization
            reg = 0.0
        else:
            # W:[layer_num,]
            W = tf.get_variable('{}_ELMo_W'.format(name),
                shape=(n_lm_layers, ),
                initializer=tf.zeros_initializer,
                regularizer=_l2_regularizer,
                trainable=True,
            )

            # normalize the weights
            # normed_weights:[layer_num,]
            normed_weights = tf.split(
                value=tf.nn.softmax(W + 1.0 / n_lm_layers),
                num_or_size_splits=n_lm_layers
            )
            # split LM layers
            # lm_embeddings: [batch, layer_num, unroll_steps, projection_dim*2]
            # layers:list of [batch, 1 , unroll_steps, projection_dim*2]
            layers = tf.split(lm_embeddings, n_lm_layers, axis=1)
    
            # compute the weighted, normalized LM activations
            layer_pieces = []
            # 将n个layer加权求和
            # normed_weights:[layer_num,]
            # layers:list of [batch, 1 , unroll_steps, projection_dim*2]
            for w, t in zip(normed_weights, layers):
                sequeeze_layer = tf.squeeze(t, squeeze_dims=1)
                if do_layer_norm:
                    layer_pieces.append(w * _do_layer_normalization(sequeeze_layer))
                else:
                    layer_pieces.append(w * sequeeze_layer)
            weighted_layer = tf.add_n(layer_pieces)
    
            # get the regularizer 
            reg = [
                r for r in tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
                if r.name.find('{}_ELMo_W/'.format(name)) >= 0
            ]
            if len(reg) != 1:
                raise ValueError

        # scale the weighted sum by gamma
        gamma = tf.get_variable(
            '{}_ELMo_gamma'.format(name),
            shape=(1, ),
            initializer=tf.ones_initializer,
            regularizer=None,
            trainable=True,
        )
        weighted_lm_layers = weighted_layer * gamma

        ret = {'weighted_op': weighted_lm_layers,
               'regularization_op': reg
               }

    return ret

