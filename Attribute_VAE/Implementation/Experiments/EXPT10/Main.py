# Dependencies
import tensorflow as tf
import numpy as np
import MNIST_Data_Loader as mnist 
import Capsule_Variational_Auto_Encoder as cvae
from datetime import datetime as dt
import os, sys, re
import time
import cv2


# Define parameters
ATTR_CAP_NUM = 0
SLACK_CAP_NUM = 7
ENC_DIM = 100
LATENT_DIM = 5
IN_DIM = 28*28
OUT_DIM = 28*28
ATTR_LR = 1e-3
LOSS_LR = 1e-3
TRAINING_ITR = 750000 + 1
RECORDING_ITR = 100
VALIDATION_ITR = 1000
TR_BATCH_SIZE = 64
VAL_BATCH_SIZE = 100


# Create an instance of a data loader
data_loader = mnist.dataLoaderForAttributedMNIST() #TODO


# Create a capsule variational auto encoder
net = cvae.capsuleVariationalAutoEncoder(attr_cap_num = ATTR_CAP_NUM, slack_cap_num = SLACK_CAP_NUM, in_dim = IN_DIM, enc_dim = ENC_DIM, latent_dim = LATENT_DIM, out_dim = OUT_DIM)


# Define the net, optimization
X = tf.placeholder(tf.float32, shape = [None, IN_DIM], name = 'X')
input_list, attr_input_list, noise_input_list, attr_pred_list, attr_pred_quant_list, cap_output_list, cap_contrib_list, cap_gen_list, cap_gen_contrib_list, KL_loss, attr_loss, output_net, gen_net = net.BuildNetworkArchitecture()
recn_loss = tf.reduce_sum(X*tf.log(output_net + 1e-8) + (1 - X)*tf.log(1 - output_net + 1e-8), axis = 1)
net_loss = recn_loss - KL_loss


# Define optimizers
# attr_optim = tf.train.AdamOptimizer(ATTR_LR)
loss_optim = tf.train.AdamOptimizer(LOSS_LR)
# attr_training_step = attr_optim.minimize(attr_loss)
loss_training_step = loss_optim.minimize(-net_loss)


# Training preparation
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)


# Information check of all trainable variables
train_vars = tf.trainable_variables()
print('[INFO] The list of all trainable variables : ')
for a_train_var in train_vars :
	print('[INFO] 		' + str(a_train_var.name))


# # Debug statements
# print('[DEBUG] Number of entries in input_list : ' + str(len(input_list)))
# for x in input_list :
# 	print('[DEBUG] 		' + str(x))
# # sys.exit()


# Get the time right now in a beautiful manner!!
time_now = str(dt.now())
fields = time_now.strip().split(' ')
f0 = fields[0].replace('-', '_')
f1 = fields[1].replace(':', '_')
f1 = f1.replace('.', '_')


# Create a log at the current time for the experiment
error_log_handle = open('Error_Log_' + f0 + '_'+  f1 + '.txt', 'a')
error_log_handle.write('an_iter' + ',' + 'KL_loss' + ',' + 'attr_loss' + ',' + 'recn_loss' + ',' + 'net_loss' + '\n') # CHANGE APPROPRIATELY


# Training iterations
for an_iter in range(TRAINING_ITR) :

	# # Print iteration information
	print('[INFO] Iteration : ' + str(an_iter))

	# Get train batch
	x, y_int, all_attr = data_loader.GetNextAttributedMNISTBatch(batch_size = TR_BATCH_SIZE, permissible_digits = '0,1,2,3,4,5', split = 'tr')	
	# print('[DEBUG] 		Number of attributes : ' + str(len(all_attr)))
	# print('[DEBUG] 		Shape of data input : ' + str(x.shape))

	# Create feed dict
	feed_dict = {}
	# Add X
	feed_dict[X] = x
	# Add input_list placeholders
	for i in range(len(input_list)) :
		feed_dict[input_list[i]] = x
	# Add attr_input_list placeholders
	# for i in range(len(all_attr)) :
	# 	feed_dict[attr_input_list[i]] = all_attr[i]

	# Optimize the net loss and the attribute losses
	sess.run(loss_training_step, feed_dict = feed_dict) # Silly error rectified
	# sess.run(attr_training_step, feed_dict = feed_dict)

	# If the iteration is "conducive for recording"
	if an_iter % RECORDING_ITR == 0 :
		# Get train batch
		# x, y_int, all_attr = data_loader.GetNextAttributedMNISTBatch(batch_size = VAL_BATCH_SIZE, permissible_digits = '0,1,2,3,4,5,6,7,8,9', split = 'val')	
		x, y_int, all_attr = data_loader.GetNextAttributedMNISTBatch(batch_size = VAL_BATCH_SIZE, permissible_digits = '6,7,8,9', split = 'val')	

		# Create feed dict
		feed_dict = {}
		# Add X
		feed_dict[X] = x
		# Add input_list placeholders
		for i in range(len(input_list)) :
			feed_dict[input_list[i]] = x
		# Add attr_input_list placeholders
		# for i in range(len(all_attr)) :
		# 	feed_dict[attr_input_list[i]] = all_attr[i]

		# Get the loss values
		KL_loss_val, attr_loss_val, recn_loss_val, net_loss_val = sess.run([KL_loss, attr_loss, recn_loss, net_loss], feed_dict = feed_dict)

		# Display and save results
		error_log_handle.write(str(an_iter) + ',' + str(np.mean(KL_loss_val)) + ',' + str(np.mean(attr_loss_val)) + ',' + str(np.mean(recn_loss_val)) + ',' + str(np.mean(net_loss_val)) + '\n')
		print('[TRAINING] 		' + 'an_iter : ' + str(an_iter) + ' ' + 'KL_loss : ' + str(np.mean(KL_loss_val)) + ' ' + 'attr_loss : ' + str(np.mean(attr_loss_val)) + ' ' + 'recn_loss : ' + str(np.mean(recn_loss_val)) + ' ' + 'net_loss : ' + str(np.mean(net_loss_val)))
		# time.sleep(5)

		if an_iter % VALIDATION_ITR == 0 :
			# Feed random attributes to slack variables
			for i in np.arange(ATTR_CAP_NUM, ATTR_CAP_NUM + SLACK_CAP_NUM, 1) :
				# feed_dict[attr_input_list[int(i)]] = np.random.random([VAL_BATCH_SIZE, 1])
				feed_dict[attr_input_list[int(i)]] = np.zeros([VAL_BATCH_SIZE, 1])
			# Feed uniform gaussian noise to Z_normal
			for i in range(len(noise_input_list)) :
				feed_dict[noise_input_list[i]] = np.random.normal(0.0, 1.0, size = [VAL_BATCH_SIZE, LATENT_DIM])
			# Get all the required information
			cap_gen_list_val, output_net_val = sess.run([cap_gen_list, output_net], feed_dict = feed_dict)
			
			# print('[DEBUG] The generated data shape is : ' + str(gen_net_val.shape))
			# print('[DEBUG] The length of capsule generations is : ' + str(len(cap_gen_list_val)) + ' ' + str(len(cap_gen_contrib_list_val)))
			# print('[DEBUG] The shape of the capsule generation is : ' + str(cap_gen_list_val[0].shape) + ' ' + str(cap_gen_contrib_list_val[0].shape))
			# sys.exit()

			# Save generated images
			index = 0
			for i in range(VAL_BATCH_SIZE) :
				cv2.imwrite('temp' + str(index).zfill(5) + '.jpg', np.reshape(x[i], [28, 28, 1])*255.0)
				index += 1
				cv2.imwrite('temp' + str(index).zfill(5) + '.jpg', np.reshape(output_net_val[i], [28, 28, 1])*255.0)
				index += 1
			os.system('ffmpeg -f image2 -r 1/1 -i temp%05d.jpg -vcodec mpeg4 -y ' + 'Generated_Dataset/Generated_Digits_Iteration_' + str(an_iter) + '.mp4')
			# time.sleep(0.1)
			os.system('rm -f temp*.jpg')
			# time.sleep(0.1)

			# Save capsule outputs
			for an_attr in range(ATTR_CAP_NUM + SLACK_CAP_NUM) :
				im_arr = cap_gen_list_val[an_attr]
				index = 0
				for i in range(VAL_BATCH_SIZE) :
					cv2.imwrite('temp' + str(index).zfill(5) + '.jpg', np.reshape(im_arr[i], [28, 28, 1])*255.0)
					index += 1
				os.system('ffmpeg -f image2 -r 1/1 -i temp%05d.jpg -vcodec mpeg4 -y ' + 'Generated_Dataset/Generated_Capsule_Outputs_Attribute_' + str(an_attr) + '_Iteration_' + str(an_iter) + '.mp4')
				# time.sleep(0.1)
				os.system('rm -f temp*.jpg')
				# time.sleep(0.1)


