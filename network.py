import logging
import tensorflow as tf

from tensorflow.python.training import saver as tf_saver


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

    output_layer = setup_layer(
        hidden_layers[-1].output_size, output_size, None, previous_layer
    )

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output_layer, y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=args.learning_rate).minimize(cost)

    return x, y, output_layer, cost, optimizer


def train_network(network, training_data, model_name, args):
    logging.info('Training model')

    x, y, output_layer, cost_function, optimizer = network

    init = tf.initialize_all_variables()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)

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

        logging.info('Model saved to %s' % model_name)


def evaluate(input_size, output_size, hidden_layers, args, model_name, value):
    x, _, output_layer, _, _ = setup_network(input_size, output_size, hidden_layers, args)

    saver = tf_saver.Saver()

    with tf.Session() as sess:
        saver.restore(sess, model_name)

        return sess.run([output_layer], feed_dict={x: value})
