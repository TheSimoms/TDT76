import logging
import tensorflow as tf
import numpy as np

from utils import get_sorted_labels, generate_dict_from_directory, preprocess_image


IMAGE_SIZE = 192 * 256 * 3


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


def setup_convolutional_network(input_size, output_size, args, channel_number=3):
    logging.debug('Setting up convolutional network')

    x = tf.placeholder(tf.float32, [None, input_size], name='x')
    y = tf.placeholder(tf.float32, [None, output_size], name='y')

    x_image = tf.reshape(x, [-1, 192, 256, channel_number])

    filter_size1 = 32
    num_filters1 = 16

    filter_size2 = 32
    num_filters2 = 32

    fc_size = 256

    layer1 = setup_convolutional_layer(
        x=x_image, num_input_channels=channel_number, filter_size=filter_size1,
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

    output_layer = tf.nn.softmax(layer_fc2)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2, labels=y)
    cost = tf.reduce_mean(cross_entropy)

    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

    return x, y, output_layer, cost, optimizer


def setup_network(input_size, output_size, hidden_layers, args):
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

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(tf.nn.softmax(output_layer), y))
    optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate).minimize(cost)

    return x, y, output_layer, cost, optimizer


def run_network(network, model_name, args, train=True, training_data=None, value=None,
                convolutional=False):
    if convolutional:
        labels = get_sorted_labels(generate_dict_from_directory(args.train_path))
        label_list = [0.0] * len(labels)

    with tf.Session() as sess:
        x, y, output_layer, cost_function, optimizer = network

        sess.run(tf.initialize_all_variables())

        saver = tf.train.Saver()

        if train:
            logging.info('Training model')

            number_of_batches = int(len(training_data[0]) / args.batch_size)

            for epoch in range(args.training_epochs):
                logging.info('Epoch: %d' % epoch)

                avreage_cost = 0.0

                for i in range(0, number_of_batches):
                    logging.info('Batch: %d' % i)

                    offset = i * args.batch_size

                    if convolutional:
                        batch_x = training_data[0][offset:offset + args.batch_size]
                        batch_y = training_data[1][offset:offset + args.batch_size]

                        x_ = [preprocess_image(image) for image in batch_x]
                        y_ = []

                        for image in batch_y:
                            image_labels = list(label_list)

                            for label, confidence in image:
                                image_labels[label] = confidence

                            y_.append(image_labels)
                    else:
                        x_ = training_data[0][offset:offset + args.batch_size]
                        y_ = training_data[1][offset:offset + args.batch_size]

                    loss, cost = sess.run([optimizer, cost_function], feed_dict={x: x_, y: y_})
                    avreage_cost += cost / number_of_batches

                if epoch % 1 == 0:
                    logging.info('Epoch %d; Cost %.9f' % (epoch + 1, avreage_cost))

            logging.info('Training complete')

            saver.save(sess, model_name)

            logging.debug('Model saved to %s' % model_name)
        else:
            saver = tf.train.import_meta_graph('%s.meta' % model_name)
            saver.restore(sess, model_name)

            return np.squeeze(sess.run([output_layer], feed_dict={x: value}))
