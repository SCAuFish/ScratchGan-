''' A discriminator based on BERT
    score each word given the whole sentence as context
    '''

import tensorflow as tf
import tensorflow_hub as hub

import random, copy
import numpy as np

BERT_DIR = "/home/aufish/Downloads/bert"

bert_module = hub.KerasLayer(BERT_DIR, trainable=True)

from bert import tokenization

def create_tokenizer(vocab_file, do_lower_case=False):
    return tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)

def convert_sentence_to_features(sentence, tokenizer, max_seq_len=50):
    tokens = ['[CLS]']
    tokens.extend(tokenizer.tokenize(sentence))
    if len(tokens) > max_seq_len-1:
        tokens = tokens[:max_seq_len-1]
    tokens.append('[SEP]')
    
    segment_ids = [0] * len(tokens)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)

    #Zero Mask till seq_length
    zero_mask = [0] * (max_seq_len-len(tokens))
    input_ids.extend(zero_mask)
    input_mask.extend(zero_mask)
    segment_ids.extend(zero_mask)
    
    return input_ids, input_mask, segment_ids

def convert_sentences_to_features(sentences, tokenizer, max_seq_len=50):
    all_input_ids = []
    all_input_mask = []
    all_segment_ids = []
    
    for sentence in sentences:
        input_ids, input_mask, segment_ids = convert_sentence_to_features(sentence, tokenizer, max_seq_len)
        all_input_ids.append(input_ids)
        all_input_mask.append(input_mask)
        all_segment_ids.append(segment_ids)
    
    return all_input_ids, all_input_mask, all_segment_ids


class WordPredictor(tf.keras.Model):
    # The output means, how possible the given word may fit into the blank
    def __init__(self, class_num, bert=bert_module, dropout=0.1):
        super(WordPredictor, self).__init__()
        self.bert = bert
        self.drop = tf.keras.layers.Dropout(rate=dropout, trainable=True)
        
        self.dense = tf.keras.layers.Dense(
            class_num,
            activation=None,
            kernel_initializer='glorot_uniform',
            name='word_prediction',
            trainable=True)
        
    def call(self, inputs, mask_loc):
        # When passed in, all tensors are stacked in one, split it into a list
        inputs = tf.unstack(tf.cast(inputs, tf.dtypes.int32), axis=1)

        pooled, sequential = self.bert(inputs)
        
        # select one from each batch
        s = tf.gather_nd(sequential, [(i, mask_loc[i]) for i in range(sequential.shape[0])])
        # s now has shape (batch_size * 768)
        
        x = self.drop(s)
        return self.dense(x)

tokenizer = create_tokenizer(BERT_DIR + "/assets/vocab.txt")

MASK_ID = tokenizer.convert_tokens_to_ids(['[MASK]'])[0]

def score_prediction(model, sentence, blank_loc, word):
    # Given a sentence and at which location (1-indexed) it is blank
    # return the predicted word
    ids, masks, seg_ids = convert_sentence_to_features(sentence, tokenizer)
    
    # adjust input_mask, reset the randomly selected mask and set with blank_loc
    masks[blank_loc] = 0
    
    ids[blank_loc] = MASK_ID
    
    bert_input = tf.stack([ids, masks, seg_ids])
    bert_input = tf.reshape(bert_input, (1, bert_input.shape[0], bert_input.shape[1]))
        
    
    output = model(bert_input, [blank_loc])

    word_id = tokenizer.convert_tokens_to_ids([word])[0]
    return tf.gather_nd(output, [0, word_id])

def score_sentence(model, sentence):
    # Given a sentence in words, return a tensor with same length
    # as a sentence. Each entry represent a score for the choice
    # of that word
    words  = sentence.split(" ")
    scores = []
    
    for i in range(len(words)):
        result = score_prediction(model, sentence, i+1, words[i])
        scores.append(result)
    
    return tf.convert_to_tensor(scores)