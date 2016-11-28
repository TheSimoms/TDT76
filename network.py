import logging
import pickle
import math
import tensorflow as tf
import numpy as np

from utils import log_header, preprocess_image


class Layer:
    def __init__(self, output_size, activation=tf.nn.relu):
        self.output_size = output_size
        self.activation = activation


def setup_weights(shape):
    """
    Set up weights for use in network

    :param shape: Shape of the weights
    :return: Weights
    """

    return tf.Variable(tf.truncated_normal(shape, stddev=1e-1, dtype=tf.float32))


def setup_biases(length, initial_value=0.0):
    """
    Set up biases for use in network

    :param length: Number of values
    :param initial_value: Initial value of the biases
    :return:
    """

    return tf.Variable(
        tf.constant(initial_value, shape=[length], dtype=tf.float32), trainable=True
    )


def setup_convolutional_layer(x, filter_size, input_size, num_filters, s=1, k=2,
                              use_pooling=True):
    """
    Set up layer for use in convolutional network

    :param x: Input layer
    :param filter_size: Filter size
    :param input_size: Number of layer input values
    :param num_filters: Number of filters
    :param s: Stride size
    :param k: Depth
    :param use_pooling: Whether to pool the layer
    :return: Convolutional layer
    """

    # Shape of layer
    shape = [filter_size, filter_size, input_size, num_filters]

    # Set up layer weights and biases
    weights = setup_weights(shape)
    biases = setup_biases(num_filters)

    # Construct the layer and add biases
    layer = tf.nn.conv2d(input=x, filter=weights, strides=[1, s, s, 1], padding='SAME')
    layer += biases

    # Activate pooling if wanted. If not, apply the ReLU function
    if use_pooling:
        layer = tf.nn.max_pool(
            value=layer, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME'
        )
    else:
        layer = tf.nn.relu(layer)

    return layer


def flatten_layer(layer):
    """
    Flatten a convolutional layer

    :param layer: Layer to flatten
    :return: Flattened layer
    """

    # Get layer shape
    layer_shape = layer.get_shape()

    # Extract number of features and flatten layer
    number_of_features = layer_shape[1:4].num_elements()
    layer_flat = tf.reshape(layer, [-1, number_of_features])

    return layer_flat, number_of_features


def setup_fully_connected_layer(x, input_size, output_size, use_relu=True):
    """
    Set up fully connected layer for use in convolutional network

    :param x: Input layer
    :param input_size: Number of layer input values
    :param output_size: Number of layer output values
    :param use_relu: Whether to use the ReLU activation function
    :return: Fully connected layer layer
    """

    # Set up weights and biases
    weights = setup_weights([input_size, output_size])
    biases = setup_biases(output_size, 1.0)

    # Construct layer
    layer = tf.matmul(x, weights) + biases

    # Apply the ReLU function if wanted
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer


def setup_layer(x, input_size, output_size, activation=None):
    """
    Set up layer for use in feedforward network

    :param x: Input layer
    :param input_size: Number of layer input values
    :param output_size: Number of layer output values
    :param activation: Layer activation function
    :return: Convolutional layer
    """

    # Set up and add weights and biases, multiply weights and input values, and add biases
    output = tf.add(
        tf.matmul(x, setup_weights([input_size, output_size])),
        setup_biases(output_size)
    )

    # Return layer if noe activation function is supplied
    if activation is None:
        return output

    # Return activated layer
    return activation(output)


def setup_convolutional_network(input_size, output_size, args):
    """
    Set up and return convolutional network

    :param input_size: Number of layer input values
    :param output_size: Number of layer output values
    :param args: Run-time arguments
    :return: Convolutional network
    """

    logging.debug('Setting up convolutional network')

    # Placeholders for input and output variables
    x = tf.placeholder(tf.float32, [None, input_size], name='x')
    y = tf.placeholder(tf.float32, [None, output_size], name='y')

    # Reshape image
    x_image = tf.reshape(x, [-1, args.image_size, args.image_size, args.number_of_channels])

    # Network parameters
    filter_size1 = 8
    num_filters1 = 32
    filter_size2 = 16
    num_filters2 = 128
    fc_size1 = 256
    fc_size2 = 512

    # Set up the convolutional layers
    conv_layer_1_1 = setup_convolutional_layer(
        x=x_image, filter_size=filter_size1, input_size=args.number_of_channels,
        num_filters=num_filters1, use_pooling=False
    )
    conv_layer_1_2 = setup_convolutional_layer(
        x=conv_layer_1_1, filter_size=filter_size1, input_size=num_filters1,
        num_filters=num_filters1, use_pooling=True
    )
    conv_layer_2_1 = setup_convolutional_layer(
        x=conv_layer_1_2, filter_size=filter_size2, input_size=num_filters1,
        num_filters=num_filters2, use_pooling=False
    )
    conv_layer_2_2 = setup_convolutional_layer(
        x=conv_layer_2_1, filter_size=filter_size2, input_size=num_filters2,
        num_filters=num_filters2, use_pooling=True
    )

    layer_flat, num_features = flatten_layer(conv_layer_2_2)

    # Set up the fully connected layers
    full_layer_1 = setup_fully_connected_layer(
        x=layer_flat, input_size=num_features, output_size=fc_size1
    )
    full_layer_2 = setup_fully_connected_layer(
        x=full_layer_1, input_size=fc_size2, output_size=output_size
    )
    full_layer_3 = setup_fully_connected_layer(
        x=full_layer_2, input_size=output_size, output_size=output_size, use_relu=False
    )

    output_layer = tf.tanh(full_layer_3)

    # Calculate cross entropy, reduce its mean, and optimize
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=output_layer, labels=y)
    cost = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

    return x, y, tf.nn.relu(full_layer_3), cost, optimizer


def setup_network(input_size, output_size, hidden_layers, args):
    """
    Set up and return feedforward network

    :param input_size: Number of layer input values
    :param output_size: Number of layer output values
    :param hidden_layers: Hidden layers
    :param args: Run-time arguments
    :return: Feedforward network
    """

    logging.debug('Setting up network')

    # Placeholders for input and output variables
    x = tf.placeholder(tf.float32, [None, input_size], name='x')
    y = tf.placeholder(tf.float32, [None, output_size], name='y')

    # Set up first hidden layer
    previous_layer = setup_layer(
        x, input_size, hidden_layers[0].output_size, hidden_layers[0].activation
    )

    # Set up the other hidden layers
    for i in range(1, len(hidden_layers)):
        previous_layer = setup_layer(
            previous_layer, hidden_layers[i - 1].output_size, hidden_layers[i].output_size,
            hidden_layers[i].activation
        )

    # Set up the output layer
    last_layer = setup_layer(
        previous_layer, hidden_layers[-1].output_size, output_size
    )

    # Calculate softmax on the output layer
    output_layer = tf.tanh(last_layer)

    # Calculate cross entropy, reduce its mean, and optimize
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=output_layer, labels=y)
    cost = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate).minimize(cost)

    return x, y, output_layer, cost, optimizer


def run_network(network, model_name, args, train=True, training_data=None, value=None,
                generating_data=None, save_path=None):
    """
    Run a neural network. Can either train weights, evaluate input values, or generate new features
    for images.

    When generating features, the features are saved in batches to pickle files

    :param network: Network to run
    :param model_name: Name of the checkpoint file. Either for saving or loading weights
    :param args: Run-time arguments
    :param train: Whether to train or evaluate
    :param training_data: Data used for training
    :param value: Value used for evaluating
    :param generating_data: Data used for generating new features
    :param save_path: Path to save new features
    :return: Evaluated value
    """

    # Set up session
    with tf.Session() as sess:
        # Extract variables from network
        x, y, output_layer, cost_function, optimizer = network

        # Initial variables and set up saver
        sess.run(tf.initialize_all_variables())
        saver = tf.train.Saver()

        # Enable training if wanted
        if train:
            logging.info('Training model')

            # Iterate training epochs
            for epoch in range(args.training_epochs):
                log_header('Epoch: %d' % epoch)

                # Iterate batches in epoch
                for batch in range(0, args.number_of_batches):
                    logging.info('Epoch: %d. Batch: %d' % (epoch, batch))

                    # Generate training batch
                    x_, y_ = training_data[0](**training_data[1])

                    # Run the optimizer
                    sess.run([optimizer, cost_function], feed_dict={x: x_, y: y_})

            logging.info('Training complete. Saving model')

            # Save trained weights
            saver.save(sess, model_name)

            logging.debug('Model saved to %s' % model_name)
        else:
            # Import and restore trained weights
            saver = tf.train.import_meta_graph('%s.meta' % model_name)
            saver.restore(sess, model_name)

            # Evaluate value if supplied
            if value is not None:
                res = []

                # Evaluate in batches in order not to deplete memory
                for i in range(0, len(value), args.batch_size):
                    res.extend(np.squeeze(sess.run(
                        [output_layer], feed_dict={x: value[i:i+args.batch_size]}
                    )))

                return res

            # Set up batches for saving features
            save_batch_number_of_batches = 50
            save_batch_size = args.batch_size * save_batch_number_of_batches
            total_testing_data = len(generating_data)
            total_save_batches = math.ceil(total_testing_data / save_batch_size)

            # Iterate every feature saving batch
            for save_batch_number in range(total_save_batches):
                logging.error('Batch %d of %d' % (save_batch_number + 1, total_save_batches))

                # Set up batch variables
                save_batch_offset = save_batch_number * save_batch_size
                save_batch_name = '%s.%d' % (args.features, save_batch_number)
                save_batch_features = {}

                # Iterate smaller batches
                for batch_number in range(save_batch_number_of_batches):
                    batch_offset = save_batch_offset + batch_number * save_batch_number_of_batches

                    if batch_offset >= total_testing_data:
                        break

                    # Extract training batch
                    batch = generating_data[batch_offset:batch_offset+save_batch_number_of_batches]
                    inputs = []

                    # Pre-process images in batch
                    for image_path, image_id in batch:
                        inputs.append(preprocess_image('%s/%s.jpg' % (image_path, image_id), args))

                    # Generate features
                    features = np.squeeze(sess.run([output_layer], feed_dict={x: inputs}))

                    # Add features to batch
                    for i in range(len(batch)):
                        _, image_id = batch[i]

                        save_batch_features[image_id] = features[i]

                # Save batch to file
                with open(save_batch_name, 'wb') as f:
                    pickle.dump(save_batch_features, f)
