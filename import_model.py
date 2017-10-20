import tensorflow as tf

sess=tf.Session()
#First let's load meta graph and restore weights
saver = tf.train.import_meta_graph('./tmp/model.ckpt.meta')
saver.restore(sess,"./tmp/model.ckpt")

graph = tf.get_default_graph()

# for op in tf.get_default_graph().get_operations():
#     print(str(op.name))

print(tf.global_variables())
