# coding=utf-8

import os

import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.contrib import rnn

from tqdm import tqdm

#from blocks.new_modules import multihead_attention, ff, ln
from blocks.modules import multihead_attention, noam_scheme, ff, ln
from blocks.position_embedding import position_embedding
from blocks.attention import attention

class BiLSTMCL(object):

    def __init__(self, args):
        self.args = args

        self.embedding_pretrained = np.fromfile(self.args.embedding_file, dtype=np.float32).reshape((-1, self.args.embedding_dim))

        model_op = self.buildModel()
        self.x           = model_op[0]
        self.y           = model_op[1]
        self.x_len       = model_op[2]
        self.logits      = model_op[3]
        self.loss        = model_op[4]
        self.prediction  = model_op[5]
        self.global_step = model_op[6]
        self.train_op    = model_op[7]
        self.summaries   = model_op[8]
        self.logits_t    = model_op[9]

        dir_path = os.path.dirname(__file__)
        self.model_path= os.path.join(dir_path, self.args.model_path)

    def train(self, train_set, eval_set=None, teacher=None):
        ## Restore
        saver = tf.train.Saver()
        sess = tf.Session()
        writer = tf.summary.FileWriter(self.args.log_dir, sess.graph)

        if self.args.restore:
            saver.restore(sess, tf.train.latest_checkpoint(self.model_path))
        else:
            print('init all')
            sess.run(tf.global_variables_initializer())

        ## Train
        train_y_t = np.fromfile(os.path.join(self.args.build_path, 'train_y_t.bin'), dtype=np.float32).reshape((-1, self.args.num_labels))
        train_x, train_y, train_x_len = train_set
        sample_num = train_x.shape[0]
        bs = self.args.batch_size
        print(train_y_t.shape)
        print(train_x.shape)

        for epoch_id in range(self.args.epoch):
            for i in tqdm(range(sample_num//bs)):
                _, loss_val, all_summary, global_step_val = sess.run( [self.train_op, self.loss, self.summaries, self.global_step],
                    feed_dict={
                        self.x: train_x[(i*bs):(i*bs+bs)],
                        self.y: train_y[(i*bs):(i*bs+bs)],
                        self.x_len: train_x_len[(i*bs):(i*bs+bs)],
                        self.logits_t: train_y_t[(i*bs):(i*bs+bs)],
                    }
                )
                writer.add_summary(all_summary, global_step_val)

            if epoch_id % self.args.save_period == 0:
                print('loss is %f'%loss_val)
                print('saving model')
                if eval_set != None:
                    self.eval(eval_set, sess)
                save_path = os.path.join(self.model_path, 'model.ckpt')
                saver.save(sess, save_path, global_step = self.global_step, write_meta_graph=False)

        print('saving model')
        save_path = os.path.join(self.model_path, 'model.ckpt')
        saver.save(sess, save_path, global_step = self.global_step, write_meta_graph=False)

        writer.close()
        sess.close()

    def eval(self, test_set, sess=None):
        if not sess:
            sess = tf.Session()
            dir_path = os.path.dirname(__file__)
            self.model_path= os.path.join(dir_path, self.args.model_path)
            saver = tf.train.Saver()
            saver.restore(sess, tf.train.latest_checkpoint(self.model_path))
            #saver.restore(sess, "output/model.ckpt-9603")

        eval_x, eval_y, eval_x_len = test_set
        sample_num = eval_x.shape[0]
        bs = self.args.batch_size

        eval_all = [0,0]
        eval_dict = [[0,0,0] for i in range(self.args.num_labels)]

        for i in tqdm(range(sample_num//bs)):
            prediction_val = sess.run(
                self.prediction,
                feed_dict={
                    self.x: eval_x[(i*bs):(i*bs+bs)],
                    self.x_len: eval_x_len[(i*bs):(i*bs+bs)],
                }
            )
            ground_truth = eval_y[(i*bs):(i*bs+bs)].tolist()

            for j in range(bs):
                pred = prediction_val[j]
                true = ground_truth[j]
                eval_dict[pred][1] += 1
                eval_dict[true][2] += 1
                eval_all[0] += 1
                if pred == true:
                    eval_dict[pred][0] += 1
                    eval_all[1] += 1
        print('Accuracy is : ', eval_all[1] / eval_all[0])

        for label_id, eval_result in enumerate(eval_dict):
            if eval_result[1] == 0:
                eval_result[1] = 2e32
            if eval_result[2] == 0:
                eval_result[2] = 2e32
            print('Accuracy && Recall of label %d is : %f %f'%(label_id, eval_result[0]/eval_result[1], eval_result[0]/eval_result[2]))

    def freeze(self):
        from tensorflow.python.framework import graph_util

        dir_path = os.path.dirname(__file__)
        self.model_path= os.path.join(dir_path, self.args.model_path)
        pb_file = os.path.join(self.model_path, 'model.pb')

        logits = tf.identity(self.logits, 'logits')
        prediction = tf.identity(self.prediction, 'prediction')

        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, tf.train.latest_checkpoint(self.model_path))
            constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['logits', 'prediction'])

        with tf.gfile.GFile(pb_file, 'wb') as f:
            f.write(constant_graph.SerializeToString())

        print('Freezing Done')
 
    def infer(self, infer_set):
        dir_path = os.path.dirname(__file__)
        self.model_path= os.path.join(dir_path, self.args.model_path)
        pb_file = os.path.join(self.model_path, 'model.pb')

        with tf.gfile.FastGFile(pb_file, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

            all_tensor = [n.name for n in graph_def.node]
            for t in all_tensor:
                print(t)

            logits, prediction = tf.import_graph_def(graph_def,return_elements=['logits:0', 'prediction:0'])
        
        sess = tf.Session()
        x = sess.graph.get_tensor_by_name('import/input/encoder_inputs:0')
        x_len = sess.graph.get_tensor_by_name('import/input/Placeholder:0')

        eval_x, eval_y, eval_x_len = infer_set
        sample_num = eval_x.shape[0]
        bs = self.args.batch_size

        eval_all = [0,0]
        eval_dict = [[0,0,0] for i in range(self.args.num_labels)]

        for i in tqdm(range(sample_num//bs)):
            prediction_val = sess.run(
                prediction,
                feed_dict={
                    x: eval_x[(i*bs):(i*bs+bs)],
                    x_len: eval_x_len[(i*bs):(i*bs+bs)],
                }
            )
            ground_truth = eval_y[(i*bs):(i*bs+bs)].tolist()

            for j in range(bs):
                pred = prediction_val[j]
                true = ground_truth[j]
                eval_dict[pred][1] += 1
                eval_dict[true][2] += 1
                eval_all[0] += 1
                if pred == true:
                    eval_dict[pred][0] += 1
                    eval_all[1] += 1
        print('Accuracy is : ', eval_all[1] / eval_all[0])

        for label_id, eval_result in enumerate(eval_dict):
            if eval_result[1] == 0:
                eval_result[1] = 2e32
            if eval_result[2] == 0:
                eval_result[2] = 2e32
            print('Accuracy && Recall of label %d is : %f %f'%(label_id, eval_result[0]/eval_result[1], eval_result[0]/eval_result[2]))

    def buildModel(self):
        # Input
        with tf.name_scope('input'):
            # N, T_in
            x = tf.placeholder(shape=(None, self.args.max_sent_len), dtype=tf.int32, name='encoder_inputs')
            # N, C
            y = tf.placeholder(shape=(None, ), dtype=tf.int32, name='decoder_inputs')
            # N
            x_len = tf.placeholder(shape=(None,), dtype=tf.int32, name='encoder_inputs_length')

            x_mask = tf.sequence_mask(x_len, self.args.max_sent_len, dtype=tf.float32)

            batch_size = tf.shape(x)[0]

            # N, T_in
            logits_t = tf.placeholder(shape=(None, self.args.num_labels), dtype=tf.float32, name='teacher_logits')

        ## Embedding
        with tf.name_scope('embed'):
            embedding = tf.Variable(self.embedding_pretrained, trainable=False)
            #embedding = tf.get_variable('embedding', shape=(self.args.vocab_size, self.args.embedding_dim))
            e_x = tf.nn.embedding_lookup(embedding, x)
            e_x += position_embedding(e_x, self.args.max_sent_len, self.args.embedding_dim)
            if self.args.mode == 'train':
                e_x = tf.nn.dropout(e_x, self.args.keep_prob)

        ## Encoder
        with tf.name_scope('encoder'):
            ## BiLSTM
            #lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(self.args.embedding_dim, forget_bias=1.0, state_is_tuple=True)
            #lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(self.args.embedding_dim, forget_bias=1.0, state_is_tuple=True)
            #(output_fw, output_bw), (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, 
            #                                                                 lstm_bw_cell, 
            #                                                                 e_x,
            #                                                                 sequence_length=x_len,
            #                                                                 dtype=tf.float32,
            #                                                                 time_major=False,
            #                                                                 scope=None)
            ## N, E
            #encoder_output = tf.concat([state_fw, state_bw], axis=-1)[0]
            #encoder_output = ln(encoder_output)

            ## BiLSTM-Att
            #lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(self.args.embedding_dim, forget_bias=1.0, state_is_tuple=True)
            #lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(self.args.embedding_dim, forget_bias=1.0, state_is_tuple=True)
            #(output_fw, output_bw), (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, 
            #                                                                 lstm_bw_cell, 
            #                                                                 e_x,
            #                                                                 sequence_length=x_len,
            #                                                                 dtype=tf.float32,
            #                                                                 time_major=False,
            #                                                                 scope=None)
            ## N, E
            #encoder_output = tf.add(output_fw, output_bw)
            #encoder_output, alpha = attention(encoder_output)
            #encoder_output = tf.nn.dropout(encoder_output, self.args.keep_prob)
            #encoder_output = ln(encoder_output)

            # encode input into a vector

            ## multi-bilstm-att
            #bi_layer_size = 4
            ## L, E
            #encode_cell_fw = rnn.MultiRNNCell([rnn.DropoutWrapper(rnn.BasicLSTMCell(self.args.embedding_dim),
            #    self.args.keep_prob, self.args.keep_prob) for i in range(bi_layer_size)])
            #encode_cell_bw = rnn.MultiRNNCell([rnn.DropoutWrapper(rnn.BasicLSTMCell(self.args.embedding_dim),
            #    self.args.keep_prob, self.args.keep_prob) for i in range(bi_layer_size)])

            #(output_fw, output_bw), (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
            #        cell_fw = encode_cell_fw,
            #        cell_bw = encode_cell_bw,
            #        inputs = e_x,
            #        sequence_length = x_len,
            #        dtype = tf.float32,
            #        time_major = False)

            #encoder_output = tf.concat([output_fw, output_bw], -1)
            #encoder_output, alpha = attention(encoder_output)
            #encoder_output = tf.nn.dropout(encoder_output, self.args.keep_prob)
            #encoder_output = ln(encoder_output)

            ## MLP + att
            #layer_size = 8
            #encoder_output = e_x
            #for i in range(layer_size):
            #    encoder_output = tf.layers.dense(encoder_output, self.args.embedding_dim, activation='tanh', use_bias=True)
            #    encoder_output = tf.nn.dropout(encoder_output, self.args.keep_prob)

            ##encoder_output = tf.reduce_sum(encoder_output, axis=1)

            #encoder_output, alpha = attention(encoder_output)
            #encoder_output = tf.nn.dropout(encoder_output, self.args.keep_prob)
            #encoder_output = ln(encoder_output)

            # Transformer-frame + att
            layer_size = 2
            encoder_output = e_x
            for i in range(layer_size):
                with tf.variable_scope('att-%d'%i):
                    encoder_output = multihead_attention(encoder_output, encoder_output, encoder_output)
                    encoder_output = ff(encoder_output, (self.args.hidden_dim, self.args.embedding_dim))

            encoder_output, alpha = attention(encoder_output)
            encoder_output = tf.nn.dropout(encoder_output, self.args.keep_prob)
            encoder_output = ln(encoder_output)

            #encoder_output = encoder_output[:, 0, :] 
            #encoder_output = tf.layers.dense(encoder_output, self.args.embedding_dim, activation='tanh', use_bias=True)
            #encoder_output = ln(encoder_output)

            ## CNN
            #encoder_output = e_x
            #embedded_chars_expanded = tf.expand_dims(encoder_output, -1) 
            ## Create a convolution + maxpool layer for each filter size
            #pooled_outputs = []
            ##for i, filter_size in enumerate(filter_sizes):
            #filter_sizes = [3,4,5]
            #for i, filter_size in enumerate(filter_sizes):
            #    with tf.name_scope("conv-maxpool-%s" % filter_size):
            #        # Convolution Layer
            #        filter_shape = [filter_size, self.args.embedding_dim, 1, self.args.num_filters]
            #        W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
            #        b = tf.Variable(tf.constant(0.1, shape=[self.args.num_filters]), name="b")
            #        conv = tf.nn.conv2d(
            #            embedded_chars_expanded,
            #            W,  
            #            strides=[1, 1, 1, 1], 
            #            padding="VALID",
            #            name="conv")
            #        # Apply nonlinearity
            #        h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
            #        # Maxpooling over the outputs
            #        pooled = tf.nn.max_pool(
            #            h,  
            #            ksize=[1, self.args.max_sent_len - filter_size + 1, 1, 1], 
            #            strides=[1, 1, 1, 1], 
            #            padding='VALID',
            #            name="pool")
            #        pooled_outputs.append(pooled)

            ## Combine all the pooled features
            #num_filters_total = self.args.num_filters * len(filter_sizes)
            ## N, FN, FS
            #h_pool = tf.concat(pooled_outputs, 3)
            ## N, Total_EF
            #h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
            #encoder_output = h_pool_flat

        ## Decoder
        # N, C
        with tf.name_scope('decoder'):
            if self.args.mode == 'train':
                encoder_output = tf.nn.dropout(encoder_output, self.args.keep_prob)
            decoder_output = tf.layers.dense(encoder_output, self.args.num_labels, use_bias=True, kernel_initializer=xavier_initializer(), kernel_regularizer=tf.contrib.layers.l2_regularizer(0.1), bias_regularizer=tf.contrib.layers.l2_regularizer(0.1))

        ## Loss
        with tf.name_scope('loss'):
            logits = tf.nn.softmax(decoder_output, axis=-1)
            y_true = tf.one_hot(y, self.args.num_labels)

            temp1 = 10
            logits_t_temp1 = tf.nn.softmax(logits_t / temp1)

            temp2 = 20
            logits_t_temp2 = tf.nn.softmax(logits_t / temp2)

            loss_1 = -tf.reduce_mean(y_true*tf.log(logits)) 
            loss_2 = -tf.reduce_mean(logits_t_temp1*tf.log(logits/temp1)) 
            loss_3 = -tf.reduce_mean(logits_t_temp2*tf.log(logits/temp2)) 
            loss = loss_1 + loss_2 + loss_3


            prediction = tf.argmax(logits, 1)

        ## train_op
        with tf.name_scope('train'):
            global_step = tf.train.get_or_create_global_step()
            lr = noam_scheme(self.args.lr, global_step, self.args.warmup_steps)
            optimizer = tf.train.AdamOptimizer(lr)
            #optimizer = tf.train.GradientDescentOptimizer(lr)
        
            ## gradient clips
            trainable_params = tf.trainable_variables()
            gradients = tf.gradients(loss, trainable_params)
            clip_gradients, _ = tf.clip_by_global_norm(gradients, self.args.gradient_clip_num)
            train_op = optimizer.apply_gradients(zip(clip_gradients, trainable_params), global_step=global_step)
        
        # Summary
        with tf.name_scope('summary'):
            tf.summary.scalar('lr', lr)
            tf.summary.scalar('loss', loss)
            tf.summary.scalar('global_step', global_step)
            summaries = tf.summary.merge_all()

        return x, y, x_len, logits, loss, prediction, global_step, train_op, summaries, logits_t
