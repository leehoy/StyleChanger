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
	else:
		pass

def maxpool2d(x,k=2):
	return tf.nn.max_pool(x,ksize=[1,k,k,1],strides=[1,k,k,1],padding='SAME')

def Network(x,weights,biases,dropout):
	x=tf.reshape(
def main():
	parser = build_parser()
	option=parser.parse_args()
	weights={
	


if __name__=='__main__':
	main()
