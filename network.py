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
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))


def setup_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))


def setup_convolutional_layer(x, num_input_channels, filter_size, num_filters, use_pooling=True):
    shape = [filter_size, filter_size, num_input_channels, num_filters]

    weights = setup_weights(shape=shape)
    biases = setup_biases(length=num_filters)

    layer = tf.nn.conv2d(input=x, filter=weights, strides=[1, 1, 1, 1], padding='SAME')

    layer += biases

    if use_pooling:
        layer = tf.nn.max_pool(
            value=layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME'
            )

    layer = tf.nn.relu(layer)

    return layer


def flatten_layer(layer):
    layer_shape = layer.get_shape()

    number_of_features = layer_shape[1:4].num_elements()

    layer_flat = tf.reshape(layer, [-1, number_of_features])

    return layer_flat, number_of_features


def setup_fully_connected_layer(x, num_inputs, num_outputs, use_relu=True):
    weights = setup_weights(shape=[num_inputs, num_outputs])
    biases = setup_biases(length=num_outputs)

    layer = tf.matmul(x, weights) + biases

    if use_relu:
        layer = tf.nn.relu(layer)

    return layer


def setup_layer(i, o, x, a):
    output = tf.add(
        tf.matmul(
            x, tf.Variable(tf.random_normal([i, o]))
        ),
        tf.Variable(tf.random_normal([o]))
    )

    if a is None:
        return output

    return a(output)


def setup_convolutional_network(input_size, output_size, args):
    logging.debug('Setting up convolutional network')

    x = tf.placeholder(tf.float32, [None, input_size], name='x')
    y = tf.placeholder(tf.float32, [None, output_size], name='y')

    x_image = tf.reshape(x, [-1, args.image_size, args.image_size, args.number_of_channels])

    filter_size1 = 32
    num_filters1 = 16

    filter_size2 = 32
    num_filters2 = 32

    fc_size = 256

    layer1 = setup_convolutional_layer(
        x=x_image, num_input_channels=args.number_of_channels, filter_size=filter_size1,
        num_filters=num_filters1, use_pooling=True
    )

    layer2 = setup_convolutional_layer(
        x=layer1, num_input_channels=num_filters1, filter_size=filter_size2,
        num_filters=num_filters2, use_pooling=True
    )

    layer_flat, num_features = flatten_layer(layer2)

    layer_fc1 = setup_fully_connected_layer(
        x=layer_flat, num_inputs=num_features, num_outputs=fc_size, use_relu=True
    )

    layer_fc2 = setup_fully_connected_layer(
        x=layer_fc1, num_inputs=fc_size, num_outputs=output_size, use_relu=False
    )

    output_layer = tf.tanh(layer_fc2)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=output_layer, labels=y)
    cost = tf.reduce_mean(cross_entropy)

    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

    return x, y, output_layer, cost, optimizer


def setup_network(input_size, output_size, hidden_layers, args, train=False):
    logging.debug('Setting up network')

    x = tf.placeholder(tf.float32, [None, input_size], name='x')
    y = tf.placeholder(tf.float32, [None, output_size], name='y')

    previous_layer = setup_layer(
        input_size, hidden_layers[0].output_size, x, hidden_layers[0].activation
    )

    for i in range(1, len(hidden_layers)):
        previous_layer = setup_layer(
            hidden_layers[i - 1].output_size, hidden_layers[i].output_size,
            previous_layer, hidden_layers[i].activation
        )

    output_layer = setup_layer(
        hidden_layers[-1].output_size, output_size, previous_layer, None
    )

    if train:
        output_layer = tf.nn.softmax(output_layer)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(tf.nn.softmax(output_layer), y))
    optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate).minimize(cost)

    return x, y, output_layer, cost, optimizer


def run_network(network, model_name, args, train=True, training_data=None, value=None,
                testing_data=None, save_path=None):
    with tf.Session() as sess:
        x, y, output_layer, cost_function, optimizer = network

        sess.run(tf.initialize_all_variables())

        saver = tf.train.Saver()

        if train:
            logging.info('Training model')

            for epoch in range(args.training_epochs):
                log_header('Epoch: %d' % epoch)

                for batch in range(0, args.number_of_batches):
                    logging.info('Epoch: %d. Batch: %d' % (epoch, batch))

                    x_, y_ = training_data[0](**training_data[1])

                    sess.run([optimizer, cost_function], feed_dict={x: x_, y: y_})

            logging.info('Training complete. Saving model')

            saver.save(sess, model_name)

            logging.debug('Model saved to %s' % model_name)
        else:
            saver = tf.train.import_meta_graph('%s.meta' % model_name)
            saver.restore(sess, model_name)

            if value is not None:
                res = []

                for i in range(0, len(value), args.batch_size):
                    res.extend(np.squeeze(sess.run(
                        [output_layer], feed_dict={x: value[i:i+args.batch_size]}
                    )))

                return res

            save_batch_number_of_batches = 50
            save_batch_size = args.batch_size * save_batch_number_of_batches
            total_testing_data = len(testing_data)
            total_save_batches = math.ceil(total_testing_data / save_batch_size)

            for save_batch_number in range(total_save_batches):
                logging.error('Batch %d of %d' % (save_batch_number + 1, total_save_batches))

                save_batch_offset = save_batch_number * save_batch_size

                save_batch_name = '%s.%d' % (args.features, save_batch_number)
                save_batch_features = {}

                for batch_number in range(save_batch_number_of_batches):
                    batch_offset = save_batch_offset + batch_number * save_batch_number_of_batches

                    if batch_offset >= total_testing_data:
                        break

                    batch = testing_data[batch_offset:batch_offset+save_batch_number_of_batches]
                    inputs = []

                    for image_path, image_id in batch:
                        inputs.append(preprocess_image('%s/%s.jpg' % (image_path, image_id), args))

                    features = np.squeeze(sess.run([output_layer], feed_dict={x: inputs}))

                    for i in range(len(batch)):
                        _, image_id = batch[i]

                        save_batch_features[image_id] = features[i]

                with open(save_batch_name, 'wb') as f:
                    pickle.dump(save_batch_features, f)
