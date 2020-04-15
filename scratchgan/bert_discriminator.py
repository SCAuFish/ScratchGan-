''' A discriminator based on BERT
    score each word given the whole sentence as context
    '''

import tensorflow as tf
import tensorflow_hub as hub

BERT_DIR = "/home/aufish/Downloads/bert"
tf.enable_eager_execution()

bert_module = hub.KerasLayer(BERT_DIR, trainable=True)