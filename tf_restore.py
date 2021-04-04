# first method: build graph from scratch.
import tensorflow as tf

def build_graph():
    w1 = tf.Variable([1,3,10,15],name='W1',dtype=tf.float32)
    w2 = tf.Variable([3,4,2,18],name='W2',dtype=tf.float32)
    w3 = tf.placeholder(shape=[4],dtype=tf.float32,name='W3')
    w4 = tf.Variable([100,100,100,100],dtype=tf.float32,name='W4')
    add = tf.add(w1,w2,name='add')
    add1 = tf.add(add,w3,name='add1')
    return w3,add1

with tf.Session() as sess:
    # if there are ckpt file, use ckpt file to reload data, or use sess.run(init) to initialize all data.
    ckpt_state = tf.train.get_checkpoint_state('./temp/')
    if ckpt_state:
        w3,add1=build_graph()
        saver = tf.train.Saver()
        saver.restore(sess, ckpt_state.model_checkpoint_path)
    else:
        w3,add1=build_graph()
        init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
        sess.run(init_op)
        saver = tf.train.Saver()
    a = sess.run(add1,feed_dict={
            w3:[1,2,3,4]
        })
    print(a)
    saver.save(sess,'./temp/model')

# second method: build graph from meta file.
# get_tensor_by_name
def build_graph():
    w1 = tf.Variable([1,3,10,15],name='W1',dtype=tf.float32)
    w2 = tf.Variable([3,4,2,18],name='W2',dtype=tf.float32)
    w3 = tf.placeholder(shape=[4],dtype=tf.float32,name='W3')
    w4 = tf.Variable([100,100,100,100],dtype=tf.float32,name='W4')
    add = tf.add(w1,w2,name='add')
    add1 = tf.add(add,w3,name='add1')
    return w3,add1

with tf.Session() as sess:
    ckpt_state = tf.train.get_checkpoint_state('./temp/')
    if ckpt_state:
        saver = tf.train.import_meta_graph('./temp/model.meta')
        graph = tf.get_default_graph()
        w3 = graph.get_tensor_by_name('W3:0')
        add1 = graph.get_tensor_by_name('add1:0')
        saver.restore(sess, tf.train.latest_checkpoint('./temp/'))
        print(sess.run(tf.get_collection('w1')[0]))
    else:
        w3,add1=build_graph()
        init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
        sess.run(init_op)
        saver = tf.train.Saver()
    a = sess.run(add1,feed_dict={
            w3:[1,2,3,4]
        })
    print(a)
    saver.save(sess,'./temp/model')
    
# get collections
def build_graph():
    w1 = tf.Variable([1,3,10,15],name='W1',dtype=tf.float32)
    w2 = tf.Variable([3,4,2,18],name='W2',dtype=tf.float32)
    w3 = tf.placeholder(shape=[4],dtype=tf.float32,name='W3')
    w4 = tf.Variable([100,100,100,100],dtype=tf.float32,name='W4')
    add = tf.add(w1,w2,name='add')
    add1 = tf.add(add,w3,name='add1')
    tf.add_to_collection('w1','W1:0')
    tf.add_to_collection('w3',w3)
    tf.add_to_collection('add1',add1)
    return w3,add1

with tf.Session() as sess:
    ckpt_state = tf.train.get_checkpoint_state('./temp/')
    if ckpt_state:
        saver = tf.train.import_meta_graph('./temp/model.meta')
        w3 = tf.get_collection('w3')[0]
        add1 = tf.get_collection('add1')[0]
        # run init_op before restore
        saver.restore(sess, tf.train.latest_checkpoint('./temp/'))
    else:
        w3,add1=build_graph()
        init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
        sess.run(init_op)
        saver = tf.train.Saver()
    a = sess.run(add1,feed_dict={
            w3:[1,2,3,4]
        })
    print(a)
    saver.save(sess,'./temp/model')