import tensorflow as tf
from data import train_images, train_labels, test_labels, test_images

DLEN = len(train_images.data[0])
DNUM = len(train_images.data)

x = tf.placeholder(tf.float32, [None, DLEN])

W = tf.Variable(tf.zeros([DLEN, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder("float", [None, 10])

cross_entropy = -tf.reduce_sum(y_*tf.log(y))

train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)

saver = tf.train.Saver()

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(DNUM):
    batch_xs = [train_images.data[i]]
    batch_ys = [train_labels.data[i]]
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print(sess.run(accuracy, feed_dict={x: test_images.data, y_: test_labels.data}))

saver.save(sess, 'model/model')

sess.close()
