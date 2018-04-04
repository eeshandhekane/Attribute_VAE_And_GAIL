# Dependencies
import tensorflow as tf
import numpy as np
import os, sys, re


# Define a class of capsule which holds a small variational auto-encoder
class capsule(object) :
	"""
	The class builds a small variational auto encoder, the output of which is modulated by a bypass parameter.
	The bypass parameter is trained by supplying attributes while training. However, while testing we supply the values that we want.
	"""

	# Define shared variable to keep track of name repetitions
	cap_name_list = []

	# Define constructor that defines all the placeholders and variables
	def __init__(self, name, in_dim = 28*28, enc_dim = 100, latent_dim = 5, out_dim = 28*28, is_non_slack = False) :
		"""
		name : 
			Name of the capsule, must be unique
		in_dim : 
			The dimension of the input placeholder. Here, mnist image size (28*28)
		enc_dim : 
			The dimension of encoding layer
		latent_dim : 
			The dimension of latent layer
		out_dim : 
			The dimension of the output placeholder. Here, mnist image size (28*28)
		is_non_slack :
			The boolean showing whether the capsule is not meant for slack variables. If True, the capsule is meant to model an attribute
		"""

		# Check if the name is already in the list
		if name in self.cap_name_list :
			print('[ERROR] Another capsule with the name "' + str(name) + '" exists.')
			print('[ERROR] Terminating ... ')
			sys.exit()
		# Else, name the capsule and add name to the list
		self.cap_name = name
		self.cap_name_list.append(self.cap_name)
		self.is_non_slack = is_non_slack
		# Define placeholders and variables for the architecture of capsule
		self.Attr = tf.placeholder(tf.float32, shape = [None, 1], name = self.cap_name + '_Attr')
		self.Z_normal = tf.placeholder(tf.float32, shape = [None, latent_dim], name = self.cap_name + '_Z_normal')
		self.X = tf.placeholder(tf.float32, shape = [None, in_dim], name = self.cap_name + '_X')
		self.ENC_W1 = tf.Variable(tf.truncated_normal([in_dim, enc_dim], stddev = 0.1), name = self.cap_name + '_ENC_W1')
		self.ENC_B1 = tf.Variable(tf.ones([enc_dim])/10, name = self.cap_name + '_ENC_B1')
		self.ENC_W2 = tf.Variable(tf.truncated_normal([enc_dim, latent_dim], stddev = 0.1), name = self.cap_name + '_ENC_W2')
		self.ENC_B2 = tf.Variable(tf.ones([latent_dim])/10, name = self.cap_name + '_ENC_B2')
		self.SAM_mu_W1 = tf.Variable(tf.truncated_normal([latent_dim, latent_dim], stddev = 0.1), name = self.cap_name + '_SAM_mu_W1')
		self.SAM_mu_B1 = tf.Variable(tf.ones([latent_dim])/10, name = self.cap_name + '_SAM_mu_B1')
		self.SAM_logstd_W1 = tf.Variable(tf.truncated_normal([latent_dim, latent_dim], stddev = 0.1), name = self.cap_name + '_SAM_logstd_W1')
		self.SAM_logstd_B1 = tf.Variable(tf.ones([latent_dim])/10, name = self.cap_name + '_SAM_logstd_B1')
		self.DEC_W1 = tf.Variable(tf.truncated_normal([latent_dim, enc_dim], stddev = 0.1), name = self.cap_name + '_DEC_W1')
		self.DEC_B1 = tf.Variable(tf.ones([enc_dim])/10, name = self.cap_name + '_DEC_B1')
		self.DEC_W2 = tf.Variable(tf.truncated_normal([enc_dim, out_dim], stddev = 0.1), name = self.cap_name + '_DEC_W2')
		self.DEC_B2 = tf.Variable(tf.ones([out_dim])/10, name = self.cap_name + '_DEC_B2')
		self.ATTR_W1 = tf.Variable(tf.truncated_normal([enc_dim, 1], stddev = 0.1), name = self.cap_name + '_ATTR_W1')
		self.ATTR_B1 = tf.Variable(tf.ones([1])/10, name = self.cap_name + '_ATTR_B1')


	# Define a function to construct the forward pass of the capsule for training
	def BuildCapsuleArchitecture(self, pkeep = 0.75, latent_dim = 5) :
		"""
		pkeep :
			The probability of keeping neurons in dropout
		latent_dim :
			Dimension of latent representation
		"""

		# Define the encoder pass
		Y1 = tf.nn.relu(tf.add(tf.matmul(self.X, self.ENC_W1), self.ENC_B1))
		Y2 = tf.nn.dropout(Y1, pkeep)
		Y3 = tf.nn.tanh(tf.add(tf.matmul(Y2, self.ENC_W2), self.ENC_B2))
		mu = tf.add(tf.matmul(Y3, self.SAM_mu_W1), self.SAM_mu_B1)
		logstd = tf.add(tf.matmul(Y3, self.SAM_logstd_W1), self.SAM_logstd_B1)
		# Define the attribute bypass
		A = tf.nn.sigmoid(tf.add(tf.matmul(Y2, self.ATTR_W1), self.ATTR_B1)) # return
		# Define sampler
		noise = tf.random_normal([1, latent_dim])
		z = mu + tf.multiply(noise, tf.exp(0.5*logstd))
		# Define the decoder pass
		Y4 = tf.nn.relu(tf.add(tf.matmul(z, self.DEC_W1), self.DEC_B1))
		Y5 = tf.nn.dropout(Y4, pkeep)
		Y6 = tf.nn.sigmoid(tf.add(tf.matmul(Y5, self.DEC_W2), self.DEC_B2)) # return
		Y_contr = tf.multiply(Y6, A) # return
		# Loss
		cap_KL_loss = -0.5*tf.reduce_sum(1 + 2*logstd - tf.pow(mu, 2) - tf.exp(2*logstd), axis = 1) # return
		if 'Attribute' in self.cap_name :
			cap_attr_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits = A, labels = self.Attr) # return 
		# Output
		A_quant = tf.floor(A + 0.5) # return

		# Define the generation pass
		G1 = tf.nn.relu(tf.add(tf.matmul(self.Z_normal, self.DEC_W1), self.DEC_B1))
		G2 = tf.nn.dropout(G1, pkeep)
		G3 = tf.nn.sigmoid(tf.add(tf.matmul(G2, self.DEC_W2), self.DEC_B2))
		G3_contrib = tf.multiply(G3, self.Attr)

		# Return the inputs, outputs and losses
		if 'Attribute' in self.cap_name :
			return self.X, self.Attr, self.Z_normal, A, A_quant, Y6, Y_contr, G3, G3_contrib, cap_KL_loss, cap_attr_loss
		else :
			return self.X, self.Attr, self.Z_normal, A, A_quant, Y6, Y_contr, G3, G3_contrib, cap_KL_loss