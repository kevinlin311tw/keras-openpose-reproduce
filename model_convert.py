import tensorflow as tf
from tensorflow.python.framework import graph_util
from keras.models import load_model
from model import get_testing_model
from keras import backend as K
import os

def peak_binary_map(hmap, threshold=0.1):
	# find local maximum
	s = tf.shape(hmap)

	#y_left = tf.slice(hmap2, [0, 0, 0, 0], [1, height, width-1, channel])
	size = s+tf.constant([0, 0, -1, 0], dtype=tf.int32)
	y_left = tf.slice(hmap, [0, 0, 0, 0], size)
	y_left = tf.pad(y_left, [[0, 0], [0, 0], [1, 0], [0, 0] ])
	y_right = tf.slice(hmap, [0, 0, 1, 0], size)
	y_right = tf.pad(y_right, [[0, 0], [0, 0], [0, 1], [0, 0] ])

	size = s+tf.constant([0, -1, 0, 0], dtype=tf.int32)
	y_top = tf.slice(hmap, [0, 0, 0, 0], size)
	y_top = tf.pad(y_top, [[0,0], [1,0], [0,0], [0,0]])
	y_bottom = tf.slice(hmap, [0, 1, 0, 0], size)
	y_bottom = tf.pad(y_bottom, [[0,0], [0,1], [0,0], [0,0]])

	size = s + tf.constant([0, -1, -1, 0], dtype=tf.int32)
	y_top_left = tf.slice(hmap, [0, 0, 0, 0], size)
	y_top_left = tf.pad(y_top_left, [[0, 0], [1, 0], [1, 0], [0,0]])
	y_top_right = tf.slice(hmap, [0, 0, 1, 0], size)
	y_top_right = tf.pad(y_top_right, [[0, 0], [1, 0], [0, 1], [0,0]])
	y_bottom_left = tf.slice(hmap, [0, 1, 0, 0], size)
	y_bottom_left = tf.pad(y_bottom_left, [[0, 0], [0, 1], [1, 0], [0,0]])
	y_bottom_right = tf.slice(hmap, [0, 1, 1, 0], size)
	y_bottom_right = tf.pad(y_bottom_right, [[0, 0], [0, 1], [0, 1], [0,0]])

	peak_map = tf.logical_and(hmap > y_left, hmap > y_right)
	peak_map = tf.logical_and(peak_map, hmap > y_top)
	peak_map = tf.logical_and(peak_map, hmap > y_bottom)
	peak_map = tf.logical_and(peak_map, hmap > y_top_left)
	peak_map = tf.logical_and(peak_map, hmap > y_top_right)
	peak_map = tf.logical_and(peak_map, hmap > y_bottom_left)
	peak_map = tf.logical_and(peak_map, hmap > y_bottom_right)
	peak_map = tf.logical_and(peak_map, hmap > tf.ones(s, tf.float32) * threshold)

	return peak_map


# load model
print('start processing...')
keras_weights_file = './training/weights/weights.0100.h5'
keras_dat_file = 'my_model_epoch100.dat'
output_fld = 'tensorflow_model/'
num_output = 3
output_names = ['paf', 'heatmap', 'peakmap']
output_pb = os.path.join(output_fld, 'tf_pose_my_model_epoch100.pb')

# convert h5 to dat file
model = get_testing_model()
model.load_weights(keras_weights_file)
model.save(keras_dat_file)
K.clear_session()

if not os.path.isdir(output_fld):
    os.mkdir(output_fld)

print('+ 1. load keras dat model')
K.set_learning_phase(0)
net_model = load_model(keras_dat_file)

pred = [None]*num_output
pred_node_names = [None]*num_output

# add scale 
scale = tf.placeholder(tf.int32, shape=(2), name='output_size')

# output_cat = tf.concat([net_model.output[0], net_model.output[1]], axis=3)
# output_resize = tf.image.resize_bicubic(output_cat, scale)
# pred[0], pred[1] = tf.split(output_resize, [38, 19], 3)

for i in range(num_output-1):
    pred_node_names[i] = output_names[i]
    #pred[i] = tf.identity(pred[i], name=pred_node_names[i])
    #pred[i] = tf.identity(net_model.output[i], name=pred_node_names[i])
    t_resized = tf.image.resize_images(net_model.output[i], scale, method=tf.image.ResizeMethod.BICUBIC)
    if i == 0:
        t_resized = tf.image.resize_bilinear(net_model.output[i], scale)
    else:
        t_resized = tf.image.resize_bicubic(net_model.output[i], scale)
    pred[i]  = tf.identity(t_resized, name=pred_node_names[i])


pred_node_names[2] = output_names[2]
peak_map = peak_binary_map(pred[1])
pred[2] = tf.identity(peak_map, name=output_names[2])
print('output nodes names are: ', pred_node_names)

sess = K.get_session()

print('+ 2. write graph in ascii')
filename = 'only_the_graph_def.pb.ascii'
tf.train.write_graph(sess.graph.as_graph_def(), output_fld, filename, as_text=True)

print('+ 3. convert variables to constants and save')

output_graph = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), pred_node_names)
with tf.gfile.GFile(output_pb, 'wb') as f:
    f.write(output_graph.SerializeToString())

