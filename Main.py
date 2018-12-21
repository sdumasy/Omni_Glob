from preprocess import ImgLoader
import time
from models import Resnet
import Constants
import DataContainer
import tensorflow as tf
import tensorlayer as tl

ldr = ImgLoader.ImgLoader()
# ldr.create_image_records()
# ldr.rec_img()
resume = True

Dc = DataContainer.DataContainer()
# Dc.visualise_data()
tf.logging.set_verbosity(tf.logging.DEBUG)
tl.logging.set_verbosity(tf.logging.DEBUG)

train_it = Dc.train_iterator
dev_it = Dc.dev_iterator
test_it = Dc.test_iterator

def calc_metric(y, lab):
    cost = tl.cost.cross_entropy(y, lab, name='xentropy')
    correct_train_prediction = tf.equal(tf.argmax(y, 1), lab)
    acc = tf.reduce_mean(tf.cast(correct_train_prediction, tf.float32))
    return cost, acc

def train():
    sess = tf.InteractiveSession()
    sess.run([train_it.initializer, dev_it.initializer, test_it.initializer])

    train_key, train_img, train_lab = train_it.get_next()
    dev_key, dev_img, dev_lab = dev_it.get_next()
    test_key, test_img, test_lab = test_it.get_next()

    train_network = Resnet.ResModel(Constants.output_classes).create_model(train_img)
    dev_network = Resnet.ResModel(Constants.output_classes).create_model(dev_img)
    test_network = Resnet.ResModel(Constants.output_classes).create_model(test_img)

    y_train = train_network.outputs
    y_dev = dev_network.outputs
    y_test = test_network.outputs

    train_cost, train_accu = calc_metric(y_train, train_lab)
    dev_cost, dev_accu = calc_metric(y_dev, dev_lab)
    test_cost, test_accu = calc_metric(y_test, test_lab)

    train_op = tf.train.AdamOptimizer(5e-5, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False).minimize(train_cost)

    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    if resume:
        print("Load existing model " + "!" * 10)
        saver.restore(sess, Constants.saved_res_model_file + "0" + ".ckpt")

    for epoch in range(Constants.epochs):
        print("EPOCH NUM:", epoch)
        start_time = time.time()
        train_loss, train_acc, n_batch = 0, 0, 0
        for s in range(Constants.n_step_epoch):
            err, ac, _ = sess.run([train_cost, train_accu, train_op])
            train_loss += err
            train_acc += ac
            n_batch += 1

            if s % 50 == 0:
                print("Epoch:", epoch, "of", Constants.epochs, "Step number:", str(s), "of", Constants.n_step_epoch)

        print("Epoch took:", time.time() - start_time)
        print("Train loss:" , (train_loss / n_batch))
        print("Training accuracy", (train_acc / n_batch))

        save_path = saver.save(sess, Constants.saved_res_model_file + str(epoch) + ".ckpt")
        print("Model saved in path: %s" % save_path)


        dev_loss, dev_acc, n_batch = 0, 0, 0
        for _ in range(int(Constants.dev_size / Constants.batch_size)):
            err, ac = sess.run([dev_cost, dev_accu])
            dev_loss += err
            dev_acc += ac
            n_batch += 1
        print("   dev loss: %f" % (dev_loss / n_batch))
        print("   dev acc: %f" % (dev_acc / n_batch))

        test_loss, test_acc, n_batch = 0, 0, 0
        for _ in range(int(Constants.test_size / Constants.batch_size)):
            err, ac = sess.run([test_cost, test_accu])
            test_loss += err
            test_acc += ac
            n_batch += 1

        print("   test loss: %f" % (test_loss / n_batch))
        print("   test acc: %f" % (test_acc / n_batch))


train()
