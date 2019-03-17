#!/usr/bin/python
# -*- coding: utf-8 -*

import os
import argparse
import glob
import apply_bpe
import vocab
import codecs
import model

import tensorflow as tf
from tensorflow.contrib import learn
#from konlpy.tag import Kkma
from nltk.tokenize import sent_tokenize

def args_parser():
    """args parser"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_dir',
                        type=str,
                        help='train data directory',
                        required=True)
    parser.add_argument('--eval_data_dir',
                        type=str,
                        help='eval data directory',
                        required=True)
    parser.add_argument('--class_size',
                        type=int,
                        help='class count',
                        default=3)
    parser.add_argument('--batch_size',
                        type=int,
                        help='batch size',
                        default=1)
    parser.add_argument('--num_epochs',
                        type=int,
                        help='train data directory',
                        default=1)
    parser.add_argument('--max_sentence_len',
                        type=int,
                        help='max length of sentence',
                        default=100)
    parser.add_argument('--max_sentence',
                        type=int,
                        help='max sentence count in a doc',
                        default=15)
    parser.add_argument('--code_file',
                        type=str,
                        help='code file for segmentation in BPE',
                        default=None)
    parser.add_argument('--vocab_file',
                        type=str,
                        help='vacabulary file',
                        required=True)
    parser.add_argument('--vocab_size',
                        type=int,
                        help='vacabulary size',
                        required=True)
    parser.add_argument('--output_keep_prob',
                        type=float,
                        default=0.5)
    parser.add_argument('--sentence_rnn_size',
                        type=int,
                        default=100)
    parser.add_argument('--word_rnn_size',
                        type=int,
                        default=100)
    parser.add_argument('--learning_rate',
                        type=float,
                        default=0.001)
    parser.add_argument('--model_dir',
                        type=str,
                        default=None)
    parser.add_argument('--steps',
                        type=int,
                        default=None)
    parser.add_argument('--embedding_size',
                        type=int,
                        default=256)
    parser.add_argument('--cell_model',
                        type=str,
                        default="GRU")

    return parser
    
def input_fn(data_dir, batch_size,  max_sentence, max_sentence_len, num_epochs, bpe, voca, shuffle=True):
    """input function"""

    def _input_fn():
        def _transform_ids(document):
            _doc = document.decode('utf8')

            #sentence tokenizer
            _doc = _doc.rsplit('\t', 1)
            #print(_doc)
            if len(_doc) != 2:
                raise ValueError("must have a label")

            doc = _doc[0]
            label = int(_doc[1])
            doc = sent_tokenize(doc)
            if bpe != None:
                doc = list(map(bpe.segment, doc))

            #check sentence len
            doc = doc[:max_sentence]

            sent_lens = len(doc)
            if len(doc) < max_sentence:
                doc = doc + ['']*(max_sentence-len(doc))

            #check word len
            word_lens = []
            doc_ids = []
            for sent in doc:
                splited_sent = sent.split()[:max_sentence_len]
                word_lens.append(len(splited_sent))
                padded_sent = splited_sent + ['PAD']*(max_sentence_len-len(splited_sent))
                sent_ids = voca.get_ids(padded_sent) 
                doc_ids.append(sent_ids)
            return doc_ids, sent_lens, word_lens, label

        dataset = tf.contrib.data.TextLineDataset(
            glob.glob(os.path.join(data_dir, "*.txt")))

        dataset = dataset.map(lambda stmt:
                              tuple(tf.py_func(_transform_ids, [stmt], [tf.int64,tf.int64,tf.int64,tf.int64])))
        dataset = dataset.map(lambda stmt_ids, sent_lens, word_lens, label:
                              ({"inputs":stmt_ids, "sent_lens": sent_lens, "word_lens": word_lens}, label))

        if shuffle:
            dataset = dataset.shuffle(buffer_size=1000)
        dataset = dataset.batch(batch_size)
        dataset = dataset.repeat(num_epochs)

        iterator = dataset.make_one_shot_iterator()

        features, labels = iterator.get_next()
        return features, labels
    return _input_fn

def main(argv=None):
    """main"""    
    args, _ = args_parser().parse_known_args()

    tf.logging.set_verbosity(tf.logging.INFO)

    #check bpe code
    if args.code_file:
        codes = codecs.open(args.code_file, encoding='utf-8')
        bpe = apply_bpe.BPE(codes)
    else:
        bpe = None

    #load voca
    voca = vocab.Vocab(args.vocab_file)

    #train input function for an estimator
    train_input_fn = input_fn(args.train_data_dir,
                              args.batch_size,
                              args.max_sentence,
                              args.max_sentence_len,
                              args.num_epochs,
                              bpe,
                              voca)

    """
    features = train_input_fn()
    init =tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    print(sess.run(features))
    """
    # init hierarchical attention network
    hierarchical_attention_network = model.HierarchicalAttentionNetwork()
  
    # create a model function
    han_model_fn = hierarchical_attention_network.create_model_fn()

    # estimator
    estimator = tf.estimator.Estimator(model_fn=han_model_fn, 
                                       model_dir=args.model_dir, 
                                       params=args)

    # train
    estimator.train(input_fn=train_input_fn,
                    steps=args.steps)

    # eval input function
    eval_input_fn = input_fn(args.eval_data_dir,
                             args.batch_size,
                             args.max_sentence,
                             args.max_sentence_len,
                             1,
                             bpe,
                             voca,
                             shuffle=True)

    #evaluate
    accuracy_score = estimator.evaluate(input_fn=eval_input_fn)["accuracy"]
    print("\nTest Accuracy: {0:f}\n".format(accuracy_score))

if __name__ == "__main__":
    main()
