#import tensorflow as tf
#from IPython.display import display, Audio
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

import tensorflow.compat.v1 as tf

tf.enable_eager_execution()
tf.disable_v2_behavior()

# Load the graph
tf.reset_default_graph()
saver = tf.train.import_meta_graph('./trainGuit/infer/infer.meta')
graph = tf.get_default_graph()
sess = tf.InteractiveSession()
saver.restore(sess, './trainGuit/model.ckpt-653')

# Create 50 random latent vectors z
_z = (np.random.rand(50, 100) * 2.) - 1

# Synthesize G(z)
z = graph.get_tensor_by_name('z:0')
G_z = graph.get_tensor_by_name('G_z:0')
_G_z = sess.run(G_z, {z: _z})

# Play audio in notebook
#display(Audio(_G_z[0, :, 0], rate=16000))

x = np.copy(_G_z)
print(x)
#plt.plot(_G_z[0,:,0])
#plt.show()
for i in range(5):
    sf.write('./trainGuit/sampleOut' + str(i) + '.wav', x[i,:,0], 16000)