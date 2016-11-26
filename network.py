import logging
import tensorflow as tf
import numpy as np

from utils import get_number_of_images


class Layer:
    def __init__(self, output_size, activation=tf.nn.relu):
        self.output_size = output_size
        self.activation = activation


def setup_layer(input_size, output_size, activation, previous_layer):
    output = tf.add(
        tf.matmul(
            previous_layer, tf.Variable(tf.random_normal([input_size, output_size]))
        ),
        tf.Variable(tf.random_normal([output_size]))
    )

    if activation is None:
        return output

    return activation(output)


def setup_network(input_size, output_size, hidden_layers, args):
    logging.debug('Setting up network')

    x = tf.placeholder('float', [None, input_size])
    y = tf.placeholder('float', [None, output_size])

    previous_layer = setup_layer(
        input_size, hidden_layers[0].output_size, hidden_layers[0].activation, x
    )

    for i in range(1, len(hidden_layers)):
        previous_layer = setup_layer(
            hidden_layers[i - 1].output_size, hidden_layers[i].output_size,
            hidden_layers[i].activation, previous_layer
        )

    last_layer = setup_layer(
        hidden_layers[-1].output_size, output_size, None, previous_layer
    )

    output = tf.nn.softmax(last_layer)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output, y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=args.learning_rate).minimize(cost)

    return x, y, output, last_layer, cost, optimizer


def network(hidden_layers, model_name, args,
            train=True, training_data=None, value=None):
    logging.info('Training model')

    with tf.Graph().as_default():
        x, y, output, last_layer, cost_function, optimizer = setup_network(
            2048, get_number_of_images(args.train_path), hidden_layers, args
        )

        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            saver = tf.train.Saver()

            if train:
                number_of_batches = int(len(training_data) / args.batch_size)

                for epoch in range(args.training_epochs):
                    avreage_cost = 0.0

                    for i in range(0, number_of_batches):
                        offset = i * args.batch_size

                        x_ = training_data[0][offset:offset + args.batch_size]
                        y_ = training_data[1][offset:offset + args.batch_size]

                        _, cost = sess.run([optimizer, cost_function], feed_dict={x: x_, y: y_})
                        avreage_cost += cost / number_of_batches

                    if epoch % 1 == 0:
                        logging.info('Epoch %d; Cost %.9f' % (epoch + 1, avreage_cost))

                logging.info('Training complete')

                saver.save(sess, model_name)

                logging.debug('Model saved to %s' % model_name)
            else:
                saver = tf.train.import_meta_graph('%s.meta' % model_name)
                saver.restore(sess, model_name)

                return (
                    np.squeeze(sess.run([output], feed_dict={x: [value]})),
                    np.squeeze(sess.run([last_layer], feed_dict={x: [value]}))
                )
