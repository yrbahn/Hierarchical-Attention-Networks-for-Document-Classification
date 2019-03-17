import tensorflow as tf
import utils

class HierarchicalAttentionNetwork(object):
    """Hierarchical Attention Network"""
    def __init__(self):
        #check available gpus
        self.available_gpus = utils.get_available_gpus()
        self.current_gpu_index = 0
        self.total_gpu_num = len(self.available_gpus)
        self.batch_size = 0
        self.loss = None

    def _get_next_gpu(self):
        #get next gpu
        if self.total_gpu_num == 0:
            return 'cpu:0'
        else:
            self.current_gpu_index %= self.total_gpu_num
            current_gpu = self.available_gpus[self.current_gpu_index]
            self.current_gpu_index += 1
            return current_gpu

    def _set_params(self, params):
        self.params = params

    def _add_attention(self, 
                       inputs, 
                       output_size,
                       activation_fn=tf.tanh, 
                       initializer=tf.contrib.layers.xavier_initializer(), 
                       scope=None):
        #assert len(inputs.get_shape()) == 3 and inputs.get_shape()[-1].value is not Nane:
        with tf.variable_scope(scope or 'attention') as attention_scope:
            context_vector = tf.get_variable(name='context_vector',
                                             shape=[output_size],
                                             initializer=initializer,
                                             dtype=tf.float32) 
            #[batch_size, seq_length, output_size]                                 
             
            input_projection = tf.contrib.layers.fully_connected(inputs, 
                                                                 output_size,
                                                                 activation_fn=activation_fn)

            
            #u = tanh(Wh + inputs), 
            #a = exp(uT*u)/sum(exp(uT*u)
            #[batch_size, seq_length]
            attn_vector = tf.nn.softmax(tf.einsum("aij,j->ai", input_projection, context_vector))

            # s = sum(a*inputs)
            #[batch_size, output_size]
            outputs = tf.einsum("aij,ai->aj", inputs, attn_vector)
            return outputs
    
    def _get_cell_model(self):
        if self.params.cell_model == "GRU":
            return tf.contrib.rnn.GRUCell
        else:
            return tf.contrib.rnn.LSTMCell

    def _build_graph(self, mode):
        # embedding
        with tf.variable_scope("embedding") as scope:
            embedding = tf.get_variable('embedding_matrix', 
                                        initializer=tf.random_uniform([self.params.vocab_size,
                                                                      self.params.embedding_size]),
                                        dtype=tf.float32)
            
            #[batch_size(document_size), sent_size, max_sent_len, embedding_size]
            embedded_inputs = tf.nn.embedding_lookup(embedding, self.inputs) 

            #[batch_size*sent_size, max_sent_len, embedding_size]
            embedded_inputs = tf.reshape(embedded_inputs, 
                                        [self.batch_size*self.params.max_sentence, 
                                        self.params.max_sentence_len, self.params.embedding_size])

            ##[self.batch_size*self.sent_size]
            self.word_lens = tf.reshape(self.word_lens,
                                        [self.batch_size*self.params.max_sentence])

        # word rnn  
        with tf.variable_scope("WordRNNLayer"):
            if mode == tf.estimator.ModeKeys.TRAIN:
                word_fw_cell = tf.contrib.rnn.DropoutWrapper(
                    self._get_cell_model()(self.params.word_rnn_size), 
                    output_keep_prob=self.params.output_keep_prob)

                word_bw_cell = tf.contrib.rnn.DropoutWrapper(
                    self._get_cell_model()(self.params.word_rnn_size), 
                    output_keep_prob=self.params.output_keep_prob)
            else:
                word_fw_cell = self._get_cell_model()(self.params.word_rnn_size)
                word_bw_cell = self._get_cell_model()(self.params.word_rnn_size)

            word_fw_cell = tf.contrib.rnn.DeviceWrapper(word_fw_cell, device=self._get_next_gpu())
            word_bw_cell = tf.contrib.rnn.DeviceWrapper(word_bw_cell, device=self._get_next_gpu())


            word_outputs, _ = tf.nn.bidirectional_dynamic_rnn(word_fw_cell, 
                                                              word_bw_cell, 
                                                              embedded_inputs,
                                                              self.word_lens,
                                                              dtype=tf.float32)
            #[batch_size*max_sentence, max_sentence_len, rnn_size*2]
            #print(word_outputs)
            concat_word_outputs = tf.concat(word_outputs, 2)
            #print(concat_word_outputs)

        # word attention
        with tf.variable_scope("WordAttention") as scope:
            #[batch_size*sent_size, class_size]
            word_attention_outputs = self._add_attention(concat_word_outputs, 
                                                         self.params.word_rnn_size*2,
                                                         scope=scope)
            print(word_attention_outputs)
            #[batch_size, sent_size, class_size]
            sentence_inputs = tf.reshape(word_attention_outputs, 
                                        [self.batch_size, 
                                        self.params.max_sentence,
                                        self.params.word_rnn_size*2])
        # sentence rnn
        with tf.variable_scope("SentenceRNNLayer") as scope:
            if mode == tf.estimator.ModeKeys.TRAIN:
                sent_fw_cell = tf.contrib.rnn.DropoutWrapper(
                    self._get_cell_model()(self.params.sentence_rnn_size), 
                    output_keep_prob=self.params.output_keep_prob)

                sent_bw_cell = tf.contrib.rnn.DropoutWrapper(
                        self._get_cell_model()(self.params.sentence_rnn_size), 
                        output_keep_prob=self.params.output_keep_prob)
            else:
                sent_fw_cell = self._get_cell_model()(self.params.sentence_rnn_size)
                sent_bw_cell = self._get_cell_model()(self.params.sentence_rnn_size)

            sent_fw_cell = tf.contrib.rnn.DeviceWrapper(sent_fw_cell, device=self._get_next_gpu())
            sent_bw_cell = tf.contrib.rnn.DeviceWrapper(sent_bw_cell, device=self._get_next_gpu())

            sentence_outputs, _ = tf.nn.bidirectional_dynamic_rnn(sent_fw_cell,
                                                                  sent_bw_cell,
                                                                  sentence_inputs,
                                                                  self.sentence_lens,
                                                                  dtype=tf.float32) 
            concat_sentence_outputs = tf.concat(sentence_outputs,2)

        # sentence attention
        with tf.variable_scope("SentenceAttention") as scope:
            #[batch_size, rnn_size]
            sent_att_outputs = self._add_attention(concat_sentence_outputs,
                                                   self.params.sentence_rnn_size*2,
                                                   scope=scope)

        with tf.variable_scope("classifierLayer") as scope:
            self.logits = tf.contrib.layers.fully_connected(sent_att_outputs, 
                                                            self.params.class_size,
                                                            activation_fn=None)
    
            #self.logits = utils.tf_print(self.logits,message="logits:")
            
    def _add_train_layer(self, mode):
        if mode != tf.estimator.ModeKeys.PREDICT:
            with tf.variable_scope("loss") as scope:
                #labels = utils.tf_print(self.labels,  message="labels:")
                #logits = utils.tf_print(self.logits,  message="logits:")
                self.loss = tf.losses.sparse_softmax_cross_entropy(labels=self.labels,
                                                                   logits=self.logits)

        if mode == tf.estimator.ModeKeys.TRAIN:
            with tf.variable_scope("trainLayer") as scope:
                self.train_op = tf.contrib.layers.optimize_loss(
                    loss=self.loss,
                    global_step=tf.contrib.framework.get_global_step(),
                    learning_rate=self.params.learning_rate,
                    optimizer="Adam")
        else:
            self.train_op = None

    def _get_predictions(self):
        self.predictions = {
            "classes": tf.argmax(input=self.logits, axis=1),
            "probabilities": tf.nn.softmax(self.logits, name="softmax_tensor"),
            "logits": self.logits
        }
        return self.predictions

    def create_model_fn(self):
        """create model function for estimator"""
        def model_fn(features, labels, params, mode):
            """model function"""
            #set inputs
            #[batch_size(document_size), sent_size, word_size]
            self.inputs = features["inputs"]
            #self.inputs = tf.Print(self.inputs, [self.inputs], message="inputs:")

            self.batch_size = tf.shape(self.inputs)[0]

            #[batch_size]
            self.sentence_lens = features["sent_lens"]

            #[batch_size*sent_size]
            self.word_lens = features["word_lens"]

            #[batch_size]
            self.labels = labels

            self._set_params(params)
            self._build_graph(mode)
            self._add_train_layer(mode)

            predictions = self._get_predictions()
 
            eval_metric_ops = {
                "accuracy": tf.metrics.accuracy(
                    labels=self.labels, predictions=predictions["classes"])}
                
            return tf.estimator.EstimatorSpec(mode, 
                                              predictions=predictions, 
                                              loss=self.loss, 
                                              train_op=self.train_op, 
                                              eval_metric_ops=eval_metric_ops)

        return model_fn
