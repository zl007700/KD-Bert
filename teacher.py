# coding=utf-8
import os
import numpy as np
import tensorflow as tf

from tqdm import tqdm

class TeacherNet(object):

    def __init__(self, args):
        self.args = args
        dir_path = os.path.dirname(__file__)
        model_path= os.path.join(dir_path, self.args.model_path)
        pb_file = os.path.join(model_path, 'model.pb')

        with tf.device('/cpu:0'):
            g1 = tf.Graph()

        self.sess = tf.Session(graph=g1, config=tf.ConfigProto(device_count={'cpu':0}))
        with self.sess.as_default():
            with g1.as_default():
                with tf.gfile.FastGFile(pb_file, "rb") as f:
                    graph_def = tf.GraphDef()
                    graph_def.ParseFromString(f.read())
    
                    #all_tensor = [n.name for n in graph_def.node]
                    #for t in all_tensor:
                    #    print(t)
                    self.logits, self.prediction = tf.import_graph_def(graph_def,return_elements=['logits:0', 'prediction:0'])

    def infer(self, infer_batch):
        x = self.sess.graph.get_tensor_by_name('import/input/encoder_inputs:0')
        x_len = self.sess.graph.get_tensor_by_name('import/input/Placeholder:0')
        decoder_logits = self.sess.graph.get_tensor_by_name('import/dense/BiasAdd:0')
    
        infer_x, infer_y, infer_x_len = infer_batch

        decoder_logits_val = self.sess.run(
            decoder_logits,
            feed_dict={
                x: infer_x,
                x_len: infer_x_len,
            }
        )
        #ground_truth = infer_y.tolist()
        return decoder_logits_val
    
    def preInfer(self, eval_set, tag_name):
        output_file = os.path.join(self.args.build_path, '%s_y_t.bin'%tag_name)

        eval_x, eval_y, eval_x_len = eval_set
        sample_num = eval_x.shape[0]
        bs = 8

        x = self.sess.graph.get_tensor_by_name('import/input/encoder_inputs:0')
        x_len = self.sess.graph.get_tensor_by_name('import/input/Placeholder:0')
        decoder_logits = self.sess.graph.get_tensor_by_name('import/dense/BiasAdd:0')

        all_output = []
        for i in tqdm(range(sample_num//bs)):
            decoder_logits_val = self.sess.run(
                decoder_logits,
                feed_dict={
                    x: eval_x[(i*bs):(i*bs+bs)],
                    x_len: eval_x_len[(i*bs):(i*bs+bs)],
                }
            )

            output_logits = decoder_logits_val.tolist()
            all_output.extend(output_logits)

        teacher_logits = np.array(all_output).astype(np.float32)
        teacher_logits.tofile(output_file)
        print(teacher_logits.shape)


