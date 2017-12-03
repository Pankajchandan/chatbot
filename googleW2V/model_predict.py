import tensorflow as tf
import pandas as pd
from gensim.models import Word2Vec, KeyedVectors
from preprocessing import process_predict_data
import random
from sequence2sequence.model_chat import talk

with open("save/model.tf", mode='rb') as f:
        graph_def = tf.GraphDef()
        try:
            graph_def.ParseFromString(f.read())
        except:
            print('try adding PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python ' +
                  'to environment.  e.g.:\n' +
                  'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python ipython\n' +
                  'See here for info: ' +
                  'https://github.com/tensorflow/tensorflow/issues/582')

with open('intent.txt') as f:
    intent = [(idx, l.strip()) for idx, l in enumerate(f.readlines())]


g = tf.Graph()
with tf.Session(graph=g) as sess, g.device('/cpu:0'):
    tf.import_graph_def(graph_def, name='cnn')
    names = [op.name for op in g.get_operations()]
print("initialized intent model with structure")
print()
for name in names:
    print(name)
    
# load word2vec model
model = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin",binary = 'True')  

def get_intent(text):
    global g
    global intent
    global model
    x = g.get_tensor_by_name(names[0] + ':0')
    softmax = g.get_tensor_by_name(names[-1] + ':0')
    seq_len = x.get_shape().as_list()[1]
    df = pd.DataFrame(index=[0], columns=["text"])
    df["text"][0] = text
    
    data_x = process_predict_data(df,model,seq_len)
    
    with tf.Session(graph=g) as sess:
        res = softmax.eval(feed_dict={x: data_x,'cnn/dropout/dropout/keep_prob:0': 1})#dropout keep =100%
    
    #print(res)
    return intent[res[0].argmax()][1], max(res[0])

def get_response(text, thresh):
    intent, prob = get_intent(text)
    print ("probability:",prob,"intent:",intent)
    if prob >= thresh and intent != None:
        res_file = "response/"+intent+".txt"
        file = open(res_file)
        response_list = file.read().strip()
        response_list = response_list.split("\n")
        return response_list[random.randint(0,len(response_list)-1)]
    else:
        return talk(text)



