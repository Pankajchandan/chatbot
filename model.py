import os, argparse
import pandas as pd
import tensorflow as tf
import numpy as np
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from preprocessing import preprocess_data


# parse hyperparameter file
hfile = open("hyperparameters.txt","r+")
params = hfile.read().split()
param_filt = params[14]
param_filt = param_filt.split(",")
for i in range(len(param_filt)):
    param_filt[i] = int(param_filt[i])


# define learning rate
learning_rate = float(params[2]) 
# l2 regularization params
l2_reg_lambda = float(params[5])
# no of epochs
epoch = int(params[8])
# batch size
batch_size = int(params[11])
# define the size of filters
filter_list = param_filt
# define number of filters of each filter size
num_filter = int(params[17])
# keep probability for dropout layer
keep_prob = float(params[20])


print("parameters being used: learning_rate, l2_reg_lambda, epoch, batch_size, filter_list, num_filter, keep_prob")
print("values: ",(learning_rate, l2_reg_lambda, epoch, batch_size, filter_list, num_filter, keep_prob))
print("***********************************************************************************************************")


##batch generator
def next_batch(X, Y, batch_size=100):
        """Batch generator with randomization.

        Parameters
        ----------
        batch_size : int, optional
            Size of each minibatch.

        Returns
        -------
        Xs, ys : np.ndarray, np.ndarray
            Next batch of inputs and labels (if no labels, then None).
        """
        # Shuffle each epoch
        current_permutation = np.random.permutation(range(len(X)))
        epoch_text = X[current_permutation, ...]
        if Y is not None:
            epoch_labels = Y[current_permutation, ...]

        # Then iterate over the epoch
        current_batch_idx = 0
        while current_batch_idx < len(X):
            end_idx = min(current_batch_idx + batch_size, len(X))
            this_batch = {
                'text': epoch_text[current_batch_idx:end_idx],
                'labels': epoch_labels[current_batch_idx:end_idx]
                if Y is not None else None
            }
            current_batch_idx += batch_size
            yield this_batch['text'], this_batch['labels']



##convert into labels and store in dict
with open("intent.txt") as file:
    intent = file.read().split()
intent_dict = {}
for i, word in enumerate(intent):
    intent_dict[word] = i


# read data from datafile
df = pd.read_csv("datafile.csv", header=0, delimiter="\t", quoting=3)


# load word2vec model
model = Word2Vec.load("trainedWord2vecmodel")


# preprocess data_X
data_x = preprocess_data(df,model)
print("*************")


# onehot encode data_y
data_y = np.array(df["intent"])
for i, word in enumerate(data_y):
    data_y[i] = intent_dict[word]
data_y = np.array(data_y, dtype=np.int8)
nb_classes = len(intent_dict)
data_y = np.eye(nb_classes)[data_y]


# split into train and test
train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=0.33, random_state=42)


# define other non user input params
# initialize l2_loss as zero
l2_loss = tf.constant(0.0)
# define sequence length
sequence_length = data_x.shape[1]
# define num_features
num_feature = data_x.shape[2]
# store the weights
pooled_outputs = []


# In[15]:


# Create the input to the network.  This is a 4-dimensional tensor!
X = tf.placeholder(name='X', shape=[None,data_x.shape[1], data_x.shape[2], data_x.shape[3]], dtype=tf.float32)

# Create the output to the network.  This is our one hot encoding of 2 possible values (TODO)!
Y = tf.placeholder(name='Y', shape=[None,data_y.shape[1]], dtype=tf.float32)


print ("building network ")
for i, filter_size in enumerate(filter_list):
    with tf.variable_scope("conv/stack/{}".format(i), reuse=None):
        # initialize filter
        W = tf.get_variable(
            name='W',
            shape=[filter_size, num_feature, 1, num_filter],
            initializer=tf.contrib.layers.xavier_initializer_conv2d())
        
        # convolve w and input
        conv = tf.nn.conv2d(
            name='conv',
            input=X,
            filter=W,
            strides=[1, 1, 1, 1],
            padding='VALID')
        
        #add bias of size = out cannels
        b = tf.get_variable(
            name='b',
            shape=[num_filter],
            initializer=tf.constant_initializer(0.0))

        H = tf.nn.bias_add(
            name='H',
            value=conv,
            bias=b)
        
        # Apply nonlinearity
        H = tf.nn.relu(H, name="relu")
        
        # max pool
        pooled = tf.nn.max_pool(H,
                 ksize=[1, sequence_length - filter_size + 1, 1, 1],
                 strides=[1, 1, 1, 1],
                 padding='VALID',
                 name="pool")
        
        pooled_outputs.append(pooled)

with tf.name_scope("preFc"):
    # combine all pooled outputs
    total_filters = num_filter * len(filter_list)
    # concat all the pooled weights
    H_pool = tf.concat(pooled_outputs, 3)
    #flatten it for fully connected layer
    H_pool_flat = tf.reshape(H_pool, [-1, total_filters])

with tf.name_scope("dropout"):
    H_drop = tf.nn.dropout(H_pool_flat, keep_prob = keep_prob)

# Final (unnormalized) layer
with tf.name_scope("output"):
    W = tf.get_variable("W",
        shape=[total_filters, nb_classes],
        initializer=tf.contrib.layers.xavier_initializer())
    # add final layer bias
    b = tf.Variable(tf.constant(0.1, shape=[nb_classes]), name="b")
    # calc l2 losses
    l2_loss += tf.nn.l2_loss(W)
    l2_loss += tf.nn.l2_loss(b)
    
    # do logit = W*X+b
    logit = tf.nn.xw_plus_b(H_drop, W, b, name="scores")
    predictions = tf.nn.softmax(logit, name="predictions")


#claulate loss and optimizer
with tf.variable_scope("FCoptimize", reuse=None):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits= logit, labels=Y)
                          + l2_reg_lambda * l2_loss)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)


# calculate accuracy
correct_predictions = tf.equal(tf.argmax(predictions, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
print ("done...")
print ("************")


path='save/'
ckpt_name = 'save/model.ckpt'
fname = 'model.tf'
dst_nodes = ['output/predictions']
saver = tf.train.Saver()

# Create a session and init
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    print("training started!!")
    print("******************")

    # Now iterate over our dataset n_epoch times
    for epoch_i in range(epoch):
        this_loss = 0
        its = 0
    
        # mini batches:
        for Xs_i, ys_i in next_batch(train_x,train_y,1):
            # Note here: we are running the optimizer so
            # that the network parameters train!
            this_loss += sess.run([loss, optimizer], feed_dict={X:Xs_i, Y:ys_i})[0]
            its += 1
            #print(this_loss / its)
        print('Training loss: ', this_loss / its)
    
        # Validation (see how the network does on unseen data).
        this_accuracy = 0
        its = 0
    
        # Do our mini batches:
        for Xs_i, ys_i in next_batch(test_x,test_y,1):
            # we measure the accuracy
            #pred = sess.run(predictions, feed_dict={X:Xs_i, Y:ys_i})
            this_accuracy += sess.run(accuracy, feed_dict={X:Xs_i, Y:ys_i})
            its += 1
            #print ("prediction ",tf.argmax(pred,1).eval(session=sess))
            #print ("actual ", tf.argmax(ys_i,1).eval(session=sess))
        print('Validation accuracy for epoch {}: '.format(epoch_i+1), this_accuracy / its)
        print("---------------------------------------")

    print("***************")
    print("Training done!!")
    save_path = saver.save(sess, ckpt_name)
    print("Model saved in file: %s" % save_path)


print ("creating protobuf...")
g_1 = tf.get_default_graph()
with tf.Session(graph = g_1) as sess:
    saver = tf.train.import_meta_graph('save/model.ckpt.meta', clear_devices=True)
    saver.restore(sess, ckpt_name)
    graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, dst_nodes)
    tf.train.write_graph(tf.graph_util.extract_sub_graph(graph_def, dst_nodes), path, fname, as_text=False)
