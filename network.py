import tensorflow as tf
import tensorflow.contrib.slim as slim
import ops
import params

def attention(inputs, scope = "attention",
                is_training = True, reuse = False,
                shared_scope = "shared_attention", shared_reuse = False):
    """
        Define Attention Network
            inputs: input images
            scope: name of attetion scope
            is_training: is training process
            reuse: reuse variable of scope
            shared_scope: name of shared attetion scope
            shared_reuse: reuse variable of shared_scope
    """

    with tf.variable_scope(scope, reuse = reuse):
        net = inputs
        channel = params.encoder.channel
        net = ops.conv(net, scope = "conv1",
                dim = channel, kernel_size = [7, 7], stride = 1, pad_type = params.attention.padding,
                activation_fn = ops.leaky_relu, is_training = is_training,
                weights_initializer = params.encoder.weights_initializer)
        for i in range(1, params.encoder.n_enconder):
            channel *= 2
            net = ops.conv(net, scope = "conv_{}".format(i + 1),
                    dim = channel, kernel_size = [3, 3], stride = 2, pad_type = params.attention.padding,
                    activation_fn = ops.leaky_relu, is_training = is_training,
                    weights_initializer = params.encoder.weights_initializer)

        for i in range(params.encoder.n_resblock):
            net = ops.resblock(net, scope = "resblock_{}".format(i + 1),
                    dim = channel, kernel_size = [3, 3], stride = 1, pad_type = params.attention.padding,
                    norm_fn = params.encoder.norm_fn, is_training = is_training,
                    weights_initializer = params.encoder.weights_initializer,
                    dropout_ratio = params.encoder.dropout_ratio)

        channel = params.decoder.channel
        for i in range(params.decoder.n_resblock):
            net = ops.resblock(net, scope = "deresblock_{}".format(params.decoder.n_resblock - i),
                    dim = channel, kernel_size = [3, 3], stride = 1, pad_type = params.attention.padding,
                    norm_fn = params.encoder.norm_fn, is_training = is_training,
                    weights_initializer = params.encoder.weights_initializer,
                    dropout_ratio = params.encoder.dropout_ratio)

        for i in range(1, params.decoder.n_decoder):
            channel = channel / 2
            net = ops.deconv(net, scope = "deconv_{}".format(params.decoder.n_decoder - i + 1),
                    dim = channel, kernel_size = [3, 3], stride = 2, pad_type = params.attention.padding,
                    activation_fn = ops.leaky_relu, is_training = is_training,
                    weights_initializer = params.decoder.weights_initializer)
        net = ops.deconv(net, scope = "deconv_1",
                    dim = 1, kernel_size = [1, 1], stride = 1, pad_type = params.attention.padding,
                    activation_fn = ops.sigmoid, is_training = is_training,
                    weights_initializer = params.decoder.weights_initializer)
        return net

def encoder(inputs, scope = "encoder",
                is_training = True, reuse = False,
                shared_scope = "shared_encoder", shared_reuse = False):

    """
        Define Encoder Network
            inputs: input images
            scope: name of encoder scope
            is_training: is training process
            reuse: reuse variable of scope
            shared_scope: name of shared encoder scope
            shared_reuse: reuse variable of shared_scope
    """

    with tf.variable_scope(scope, reuse = reuse):
        channel = params.encoder.channel
        net = inputs
        net = ops.conv(net, scope = "conv1",
                dim = channel, kernel_size = [7, 7], stride = 1, pad_type = params.encoder.padding,
                activation_fn = ops.leaky_relu, is_training = is_training,
                weights_initializer = params.encoder.weights_initializer)
        for i in range(1, params.encoder.n_enconder):
            channel *= 2
            net = ops.conv(net, scope = "conv_{}".format(i + 1),
                    dim = channel, kernel_size = [3, 3], stride = 2, pad_type = params.encoder.padding,
                    activation_fn = ops.leaky_relu, is_training = is_training,
                    weights_initializer = params.encoder.weights_initializer)
        for i in range(params.encoder.n_resblock - 1):
            net = ops.resblock(net, scope = "resblock_{}".format(i + 1),
                    dim = channel, kernel_size = [3, 3], stride = 1, pad_type = params.encoder.padding,
                    norm_fn = params.encoder.norm_fn, is_training = is_training,
                    weights_initializer = params.encoder.weights_initializer,
                    dropout_ratio = params.encoder.dropout_ratio)

    with tf.variable_scope(shared_scope, reuse = shared_reuse):
        chanel = params.decoder.channel
        net = ops.resblock(net, scope = "resblock_{}".format(params.encoder.n_resblock),
                    dim = channel, kernel_size = [3, 3], stride = 1, pad_type = params.encoder.padding,
                    norm_fn = params.encoder.norm_fn, is_training = is_training,
                    weights_initializer = params.encoder.weights_initializer,
                    dropout_ratio = params.encoder.dropout_ratio)
    with tf.variable_scope(scope, reuse = reuse):
        net = ops.gaussian_noise_layer(net)
        return net

def decoder(inputs, scope = "decoder",
                is_training = True, reuse = False,
                shared_scope = "shared_decoder", shared_reuse = False):
    """
        Define Decoder Network
            inputs: input images
            scope: name of decoder scope
            is_training: is training process
            reuse: reuse variable of scope
            shared_scope: name of shared decoder scope
            shared_reuse: reuse variable of shared_scope
    """
    with tf.variable_scope(shared_scope, reuse = shared_reuse):
        net = inputs
        channel = params.decoder.channel
        net = ops.resblock(net, scope = "resblock_{}".format(params.encoder.n_resblock),
                    dim = channel, kernel_size = [3, 3], stride = 1, pad_type = params.decoder.padding,
                    norm_fn = params.encoder.norm_fn, is_training = is_training,
                    weights_initializer = params.encoder.weights_initializer,
                    dropout_ratio = params.encoder.dropout_ratio)

    with tf.variable_scope(scope, reuse = reuse):
        channel = params.decoder.channel
        for i in range(1, params.decoder.n_resblock):
            net = ops.resblock(net, scope = "resblock_{}".format(params.decoder.n_resblock - i),
                    dim = channel, kernel_size = [3, 3], stride = 1, pad_type = params.decoder.padding,
                    norm_fn = params.encoder.norm_fn, is_training = is_training,
                    weights_initializer = params.encoder.weights_initializer,
                    dropout_ratio = params.encoder.dropout_ratio)

        for i in range(1, params.decoder.n_decoder):
            channel = channel / 2
            net = ops.deconv(net, scope = "deconv_{}".format(params.decoder.n_decoder - i + 1),
                    dim = channel, kernel_size = [3, 3], stride = 2, pad_type = params.decoder.padding,
                    activation_fn = ops.leaky_relu, is_training = is_training,
                    weights_initializer = params.decoder.weights_initializer)
        net = ops.deconv(net, scope = "deconv_1",
                    dim = 3, kernel_size = [1, 1], stride = 1, pad_type = params.decoder.padding,
                    activation_fn = ops.tanh, is_training = is_training,
                    weights_initializer = params.decoder.weights_initializer)
        return net

def discriminator(inputs, scope = "discriminator",
                    is_training = True, reuse = False,
                    shared_scope = "shared_discriminator", shared_reuse = False):
    """
        Define Discriminator Network
            inputs: input images
            scope: name of discriminator scope
            is_training: is training process
            reuse: reuse variable of scope
            shared_scope: name of shared discriminator scope
            shared_reuse: reuse variable of shared_scope
    """

    with tf.variable_scope(scope, reuse = reuse):
        net = inputs
        channel = params.discriminator.channel
        for i in range(params.discriminator.n_discriminator - 1):
            net = ops.conv(net, scope = "conv_{}".format(i + 1),
                    dim = channel, kernel_size = [3, 3], stride = 2, pad_type = params.discriminator.padding,
                    activation_fn = ops.leaky_relu, is_training = is_training,
                    weights_initializer = params.discriminator.weights_initializer)
            channel *= 2

        net = ops.conv(net, scope = "conv_6",
                    dim = 1, kernel_size = [1, 1], stride = 1, pad_type = params.discriminator.padding,
                    activation_fn = None, is_training = is_training,
                    weights_initializer = params.discriminator.weights_initializer)

        #Final Conv Layer uses Sigmoid
        return net

def multiscale_discriminator(inputs, scope = "discriminator",
                                is_training = True, reuse = False,
                                shared_scope = "shared_discriminator", shared_reuse = False):
    nets = []
    inputs_scale = inputs
    for scale in xrange(params.discriminator.n_scales):
        net = discriminator(inputs_scale, "{}_scale_{}".format(scope, scale + 1),
                                is_training = is_training, reuse = reuse,
                                shared_scope = "{}_scale_{}".format(shared_scope, scale + 1), shared_reuse = shared_reuse)
        inputs_scale = ops.add_padding(inputs_scale, kernel_size = 3, pad_type = params.discriminator.padding)
        inputs_scale = slim.max_pool2d(inputs_scale, [3, 3], stride = 2, padding = "VALID")
        nets.append(net)
    return nets

