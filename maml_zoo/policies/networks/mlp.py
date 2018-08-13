import tensorflow as tf
from maml_zoo.utils.utils import get_original_tf_name, get_last_scope


def create_mlp(name,
               output_dim,
               hidden_sizes,
               hidden_nonlinearity,
               output_nonlinearity,
               input_dim=None,
               input_var=None,
               w_init=tf.contrib.layers.xavier_initializer(),
               b_init=tf.zeros_initializer(),
               reuse=False
               ):
    """
    Creates a MLP network
    Args:
        name (str): scope of the neural network
        output_dim (int): dimension of the output
        hidden_sizes (tuple): tuple with the hidden sizes of the fully connected network
        hidden_nonlinearity (tf): non-linearity for the activations in the hidden layers
        output_nonlinearity (tf or None): output non-linearity. None results in no non-linearity being applied
        input_dim (tuple): dimensions of the input variable e.g. (None, action_dim)
        input_var (tf.placeholder or tf.Variable or None): Input of the network as a symbolic variable
        w_init (tf.initializer): initializer for the weights
        b_init (tf.initializer): initializer for the biases
        reuse (bool): reuse or not the network

    Returns:
        input_var (tf.placeholder or tf.Variable): Input of the network as a symbolic variable
        output_var (tf.Tensor): Output of the network as a symbolic variable

    """

    assert input_var is not None or input_dim is not None

    if input_var is None:
        input_var = tf.placeholder(dtype=tf.float32, shape=input_dim, name='input')
    with tf.variable_scope(name):
        x = input_var

        for idx, hidden_size in enumerate(hidden_sizes):
            x = tf.layers.dense(x,
                                hidden_size,
                                name='hidden_%d' % idx,
                                activation=hidden_nonlinearity,
                                kernel_initializer=w_init,
                                bias_initializer=b_init,
                                reuse=reuse,
                                )

        output_var = tf.layers.dense(x,
                                     output_dim,
                                     name='output',
                                     activation=output_nonlinearity,
                                     kernel_initializer=w_init,
                                     bias_initializer=b_init,
                                     reuse=reuse,
                                     )

    return input_var, output_var


def forward_mlp(output_dim,
                hidden_sizes,
                hidden_nonlinearity,
                output_nonlinearity,
                input_var,
                mlp_params,
                ):
    """
    Creates the forward pass of an mlp given the input vars and the mlp params. Assumes that the params are passed in
    order i.e. [hidden_0/kernel, hidden_0/bias, hidden_1/kernel, hidden_1/bias, ..., output/kernel, output/bias]
    Args:
        output_dim (int): dimension of the output
        hidden_sizes (tuple): tuple with the hidden sizes of the fully connected network
        hidden_nonlinearity (tf): non-linearity for the activations in the hidden layers
        output_nonlinearity (tf or None): output non-linearity. None results in no non-linearity being applied
        input_var (tf.placeholder or tf.Variable): Input of the network as a symbolic variable
        mlp_params (list): List of the params of the neural network. It assumes that the params are passed in
        order i.e. [hidden_0/kernel, hidden_0/bias, hidden_1/kernel, hidden_1/bias, ..., output/kernel, output/bias]

    Returns:
        input_var (tf.placeholder or tf.Variable): Input of the network as a symbolic variable
        output_var (tf.Tensor): Output of the network as a symbolic variable

    """
    x = input_var
    idx = 0
    bias_added = False
    sizes = tuple(hidden_sizes) + (output_dim,)

    if output_nonlinearity is None:
        output_nonlinearity = tf.identity

    for param in mlp_params:
        name, scope = get_original_tf_name(param.name), get_last_scope(param.name)

        assert str(idx) in scope or (idx == len(hidden_sizes) and "output" in scope)

        if "kernel" in name:
            assert param.shape == (x.shape[-1], sizes[idx])
            x = tf.matmul(param, x)
        elif "bias" in name:
            assert param.shape == (sizes[idx],)
            x = tf.add(x, param)
            bias_added = True
        else:
            raise NameError

        if bias_added:
            if "hidden" in scope:
                x = hidden_nonlinearity(x)
                idx += 1
            elif "output" in scope:
                x = output_nonlinearity(x)
                idx += 1
            else:
                raise NameError
            idx += 1
            bias_added = False
    output_var = x
    return input_var, output_var

