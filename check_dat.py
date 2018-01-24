import tensorflow as tf
from tensorflow.python.framework import graph_util
from model import get_testing_model
from keras.models import load_model
from keras import backend as K
import os
import cv2
import numpy as np
import code

keras_model_file = "my_model_epoch20.dat"
output_fld = 'tensorflow_model/'
num_output = 2
output_names = ['paf', 'heatmap']
output_pb = os.path.join(output_fld, 'tf_pose_my_model_epoch20.pb')

if not os.path.isdir(output_fld):
    os.mkdir(output_fld)

# Load image
test_fname = './sample_images/ski.jpg'
test_image = cv2.imread(test_fname)
height, width, depth = test_image.shape
data = test_image[np.newaxis, ...]


# Run model with Keras
'''
net_model = load_model(keras_model_file)
output_blobs = net_model.predict(data)
code.interact(local=locals())
'''

# Print layers
'''
inputs = net_model.input                                           # input placeholder
out = [layer.output for layer in net_model.layers]          # all layer outputs
print inputs
print out
'''

# Run model with Tensorflow

with tf.Session() as sess:

	print('+ 1. load keras model')
	# Load model
	K.set_learning_phase(0)
	net_model = load_model(keras_model_file)
	graph = tf.get_default_graph()
	# for op in tf.get_default_graph().get_operations():
	# 	print str(op.name) 
	output_paf = graph.get_tensor_by_name('Mconv7_stage6_L1/BiasAdd:0')
	output_heatmap = graph.get_tensor_by_name('Mconv7_stage6_L2/BiasAdd:0')
	input_image = graph.get_tensor_by_name('input_1:0')

	paf, heatmap = sess.run( [output_paf, output_heatmap], feed_dict={input_image: data})
	print np.max(heatmap[:,:,:,5])
	print np.max(heatmap[:,:,:,4])
	print np.max(heatmap[:,:,:,2])
	code.interact(local=locals())


# Convert dat to pb
'''
print('+ 1. load keras model')
net_model = load_model(keras_model_file)

pred = [None]*num_output
pred_node_names = [None]*num_output
for i in range(num_output):
    pred_node_names[i] = output_names[i]
    pred[i] = tf.identity(net_model.output[i], name=pred_node_names[i])
print('output nodes names are: ', pred_node_names)

sess = K.get_session()

print('+ 2. write graph in ascii')
filename = 'only_the_graph_def.pb.ascii'
tf.train.write_graph(sess.graph.as_graph_def(), output_fld, filename, as_text=True)

print('+ 3. convert variables to constants and save')
output_graph = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), pred_node_names)
with tf.gfile.GFile(output_pb, 'wb') as f:
    f.write(output_graph.SerializeToString())
'''

