import tensorflow as tf
import numpy as np
import sys,os

def build_parser():
	pass

def conv2d(x,W,b,stride=1,activation='ReLU'):
	x=tf.nn.conv2d(x,W,strides=[1,stride,stride,1],padding='SAME')
	x=tf.nn.bias_add(x,b)
	if activation=='ReLU':
		return tf.nn.relu(x)
	elif activation=='sigmoid':
		pass
	elif activation=='tanh':
		pass
	elif activation=='None':
		return x
	else:
		pass

def maxpool2d(x,k=2):
	return tf.nn.max_pool(x,ksize=[1,k,k,1],strides=[1,k,k,1],padding='SAME')

def VGGNet_19(x,weights,biases,dropout,options):
	
	conv1=conv2d(x,weights['w1'],biases['b1'],1,'ReLU')
	conv2=conv2d(conv1,weights['w2'],biases['b2'],1,'ReLU')
	conv2=maxpool2d(conv2,2)
	conv3=conv2d(conv2,weights['w3'],biases['b3'],1,'ReLU')
	conv4=conv2d(conv3,weights['w4'],biases['b4'],1,'ReLU')
	conv4=maxpool2d(conv4,2)
	conv5=conv2d(conv4,weights['w5'],biases['b5'],1,'ReLU')
	conv6=conv2d(conv5,weights['w6'],biases['b6'],1,'ReLU')
	conv7=conv2d(conv6,weights['w7'],biases['b7'],1,'ReLU')
	conv8=conv2d(conv7,weights['w8'],biases['b8'],1,'ReLU')
	conv8=maxpool2d(conv7,2)
	conv9=conv2d(conv8,weights['w9'],biases['b9'],1,'ReLU')
	conv10=conv2d(conv9,weights['w10'],biases['b10'],1,'ReLU')
	conv11=conv2d(conv10,weigths['w11'],biases['b11'],1,'ReLU')
	conv12=conv2d(conv11,weigths['w12'],biases['b12'],1,'ReLU')
	conv13=conv2d(conv12,weights['w13'],biases['b13'],1,'ReLU')
	
	fc1=tf.reshape(conv13)
	fc2=tf.add(tf.matmul(fc1,weights['wd2'],baises['bd2']))
	
def VGGNet_16(x,weights,biases,dropout,options):
	pass

def style_loss(sess,model):
	
	pass

def content_loss(sess,model):
	
	pass

IMAGE_WIDTH=248
IMAGE_HEIGHT=248

learning_rate=0.1
total_loss=style_weight*style_loss(sess,model)+contents_weight*content_loss(sess,model)
optimizer=tf.train.AdamOptimizer(2.0)
train_step=optimizer.minimize(total_loss)
init=tf.global_variables_initialize()
with tf.Session() as sess:
	sess.run(init)
	sess.
def train():
	pass
	# The network has two loss functions
	# one is style loss and the other is contents loss
	# 
	contents_image=np.fromfile(contents_image_filename)
	
	weights={
	


if __name__=='__main__':
	main()
