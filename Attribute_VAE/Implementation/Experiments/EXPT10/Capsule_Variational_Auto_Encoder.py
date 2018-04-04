# Dependencies
import tensorflow as tf
import numpy as np
import os, sys, re
from Capsule import capsule


# Define a class that holds the capsule variational auto encoder
class capsuleVariationalAutoEncoder(object) :
	"""
	This class holds the entire architecture of the capsule variational auto encoder
	It creates capsules, maintains their losses and outputs, and performs training and testing as required
	"""


	# Define an array to hold capsules
	cap_list = []


	# Define constructor
	def __init__(self, attr_cap_num, slack_cap_num, in_dim = 28*28, enc_dim = 100, latent_dim = 5, out_dim = 28*28) :
		"""
		attr_cap_num :
			Number of capsules in the architecture that are intended to model attributes
		slack_cap_num :
			Number of capsules in the architecture that model slack attributes
		in_dim : 
			The dimension of the input placeholder. Here, mnist image size (28*28)
		enc_dim : 
			The dimension of encoding layer
		latent_dim : 
			The dimension of latent layer
		out_dim : 
			The dimension of the output placeholder. Here, mnist image size (28*28)
		"""

		print('[INFO] Initializing network architecture ...')

		# Create all the parameters of the network
		self.attr_cap_num = attr_cap_num
		self.slack_cap_num = slack_cap_num
		self.in_dim = in_dim
		self.enc_dim = enc_dim
		self.latent_dim = latent_dim
		self.out_dim = out_dim

		# Create capsules for attributes
		for i in range(attr_cap_num) :
			a_cap = capsule(name = 'Attribute_' + str(i), is_non_slack = True)
			self.cap_list.append(a_cap)
		for i in range(slack_cap_num) :
			a_cap = capsule(name = 'Slack_' + str(i), is_non_slack = False)
			self.cap_list.append(a_cap)
		print('[INFO] Attribute Capsules : ')
		for i in self.cap_list :
			if 'Attribute' in i.cap_name :
				print('[INFO] 		' + str(i.cap_name))
		print('[INFO] Slack Capsules : ')
		for i in self.cap_list :
			if 'Slack' in i.cap_name :
				print('[INFO] 		' + str(i.cap_name))

		print('[INFO] Network architecture initialized ...')


	# Define a function to build
	def BuildNetworkArchitecture(self) :

		# Define lists for loss and training
		input_list = [] # For self.X, return
		attr_input_list = [] # For self.Attr, return
		noise_input_list = [] # For self.Z_normal, return
		attr_pred_list = [] # For A, return
		attr_pred_quant_list = [] # For A_quant, return
		cap_output_list = [] # For Y6, return
		cap_contrib_list = [] # For Y_contrib, return
		cap_gen_list = [] # For G3, return
		cap_gen_contrib_list = [] # G3_contrib, return
		cap_KL_loss_list = [] # For cap_KL_loss, return
		cap_attr_loss_list = [] # For cap_attr_loss, return


		# For each capsule, process
		cap_num = len(self.cap_list)
		for i in range(cap_num) :
			a_cap = self.cap_list[i]
			if 'Attribute' in a_cap.cap_name :
				X, Attr, Z_normal, A, A_quant, Y_cap, Y_contrib, G3, G3_contrib, cap_KL_loss, cap_attr_loss = a_cap.BuildCapsuleArchitecture()
				input_list.append(X)
				attr_input_list.append(Attr)
				noise_input_list.append(Z_normal)
				attr_pred_list.append(A)
				attr_pred_quant_list.append(A_quant)
				cap_output_list.append(Y_cap)
				cap_contrib_list.append(Y_contrib)
				cap_gen_list.append(G3)
				cap_gen_contrib_list.append(G3_contrib)
				cap_KL_loss_list.append(cap_KL_loss)
				cap_attr_loss_list.append(cap_attr_loss)
			if 'Slack' in a_cap.cap_name :
				X, Attr, Z_normal, A, A_quant, Y_cap, Y_contrib, G3, G3_contrib, cap_KL_loss = a_cap.BuildCapsuleArchitecture()
				input_list.append(X)
				attr_input_list.append(Attr)
				noise_input_list.append(Z_normal)
				attr_pred_list.append(A)
				attr_pred_quant_list.append(A_quant)
				cap_output_list.append(Y_cap)
				cap_contrib_list.append(Y_contrib)
				cap_gen_list.append(G3)
				cap_gen_contrib_list.append(G3_contrib)
				cap_KL_loss_list.append(cap_KL_loss)

		# Define the output, and net loss
		KL_loss = tf.add_n(cap_KL_loss_list) # return
		# attr_loss = tf.add_n(cap_attr_loss_list) # return
		attr_loss = tf.constant([0.0], dtype = tf.float32) # return
		output_net = tf.sigmoid(tf.add_n(cap_contrib_list)) # return
		gen_net = tf.sigmoid(tf.add_n(cap_gen_contrib_list)) # return

		# Define reconstruction loss
		# recn_loss = tf.reduce_sum(self.X*tf.log(output_net + 1e-8) + (1 - self.X)*tf.log(1 - output_net + 1e-8), axis = 1) # return

		# Return all the outputs and the losses
		return input_list, attr_input_list, noise_input_list, attr_pred_list, attr_pred_quant_list, cap_output_list, cap_contrib_list, cap_gen_list, cap_gen_contrib_list, KL_loss, attr_loss, output_net, gen_net
