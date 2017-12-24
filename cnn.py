import tensorflow as tf
import basic_cnn as bcnn
import basic_ae as bae
from decorator import lazy_property


class CNN:import tensorflow as tf
import numpy as np
import basic_cnn as cnn
import os
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_boolean('train', False, 'Train and save the CNN model.')

# Placeholder nodes.
images_holder = tf.placeholder(tf.float32, [None, 784])
label_holder = tf.placeholder(tf.float32, [None, 10])
p_keep_holder = tf.placeholder(tf.float32)
rerank_holder = tf.placeholder(tf.float32, [None, 10])

mnist = input_data.read_data_sets("./MNIST_data", one_hot=True)


def main(arvg=None):
    """
    """
    if FLAGS.train:
        train()
    else:
        test()


def test():
    """
    """
    print "ok\n"
    batch_size = 64
    batch_xs, batch_ys = mnist.test.next_batch(batch_size)
    model = cnn.BasicCnn(images_holder, label_holder, p_keep_holder)
	
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        model.load(sess, './Models/AE_for_ATN/BasicCNN')
        
        print('Accuracy: {0:0.5f}'.format(
            sess.run(model.accuracy, feed_dict={
                #images_holder: mnist.test.images,
                #label_holder: mnist.test.labels,
                images_holder: batch_xs,
                label_holder: batch_ys,
                p_keep_holder: 1.0
            })))


def train():
    """
    """
    attack_target = 8
    alpha = 1.5
    training_epochs = 10
    batch_size = 64

    model = cnn.BasicCnn(images_holder, label_holder, p_keep_holder)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        total_batch = int(mnist.train.num_examples/batch_size)
        #for epoch in range(training_epochs):
        for epoch in range(3):
            for i in range(total_batch):
            #for i in range(3):
            	print epoch, training_epochs, i, total_batch 
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)

                loss = sess.run(model.optimization,
                                 feed_dict={
                                     images_holder: batch_xs,
                                     label_holder: batch_ys,
                                     p_keep_holder: 1.0
                                 })
            print('Eopch {0} completed. loss = {1}'.format(epoch+1, loss))
        print("Optimization Finished!")

        model.save(sess, './Models/AE_for_ATN/BasicCNN')
        print("Trained params have been saved to './Models/AE_for_ATN/BasicCNN'")


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.app.run()
       

