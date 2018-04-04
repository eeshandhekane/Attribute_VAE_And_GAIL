# Dependencies
import os, sys, re
import tensorflow as tf
import numpy as np
import random
from tensorflow.examples.tutorials.mnist import input_data


# Parameters
CLASS_NUM_DEFAULT = 10
ATTR_NUM_DEFAULT = 7
BATCH_SIZE_DEFAULT = 64


# Define a class to load MNIST data
class dataLoaderForAttributedMNIST(object) :
	"""
	This class loads the MNIST data and loads batches with attributes
	"""


	# Constructor
	def __init__(self, class_num = 10, attr_num = 7, is_default_attr = True) :

		print('[INFO] Initializing the attributed MNIST data loader.')

		# Load MNIST
		self.mnist = input_data.read_data_sets('/tmp/data', one_hot = True)

		# Store indices of all digits 
		self.train_digit_2_indices = []
		self.validation_digit_2_indices = []
		self.test_digit_2_indices = []
		# Create digit-many lists
		for a_digit in range(10) : # 10 = self.mnist.train.labels.shape[1]
			self.train_digit_2_indices.append([])
			self.validation_digit_2_indices.append([])
			self.test_digit_2_indices.append([])
		# Distribute all indices properly for train split
		train_all_indices = np.argmax(self.mnist.train.labels, axis = 1)
		train_len = int(self.mnist.train.labels.shape[0])
		for i in range(train_len) :
			self.train_digit_2_indices[train_all_indices[i]].append(int(i))
		# Distribute all indices properly for validation split
		validation_all_indices = np.argmax(self.mnist.validation.labels, axis = 1)
		validation_len = int(self.mnist.validation.labels.shape[0])
		for i in range(validation_len) :
			self.validation_digit_2_indices[validation_all_indices[i]].append(int(i))
		# Distribute all indices properly for validation split
		test_all_indices = np.argmax(self.mnist.test.labels, axis = 1)
		test_len = int(self.mnist.test.labels.shape[0])
		for i in range(test_len) :
			self.test_digit_2_indices[test_all_indices[i]].append(int(i))

		# Show stats
		print('[INFO] 	Distribution of indices over all classes : ')
		for i in range(10) :
			print('[INFO] 		Digit : ' + str(i) + ' Train Samples : ' + str(len(self.train_digit_2_indices[i])) + ' Validation Samples : ' + str(len(self.validation_digit_2_indices[i])) + ' Testing Samples : ' + str(len(self.test_digit_2_indices[i])))
		

		# Define for each class the corresponding attribute
		self.class_num = class_num
		self.attr_num = attr_num
		self.class_2_attr = np.zeros([self.class_num, self.attr_num])

		# If using default attributes
		if is_default_attr :
			if self.class_num == CLASS_NUM_DEFAULT and self.attr_num == ATTR_NUM_DEFAULT :
				print('[INFO] 		Using default attributes for class to attributes mapping.')
				self.class_2_attr[0] = np.array([1, 1, 1, 0, 1, 1, 1]).astype(np.float32) # 0
				self.class_2_attr[1] = np.array([0, 0, 1, 0, 0, 1, 0]).astype(np.float32) # 1
				self.class_2_attr[2] = np.array([1, 0, 1, 1, 1, 0, 1]).astype(np.float32) # 2
				self.class_2_attr[3] = np.array([1, 0, 1, 1, 0, 1, 1]).astype(np.float32) # 3
				self.class_2_attr[4] = np.array([0, 1, 1, 1, 0, 1, 0]).astype(np.float32) # 4
				self.class_2_attr[5] = np.array([1, 1, 0, 1, 0, 1, 1]).astype(np.float32) # 5
				self.class_2_attr[6] = np.array([1, 1, 0, 1, 1, 1, 1]).astype(np.float32) # 6
				self.class_2_attr[7] = np.array([1, 0, 1, 0, 0, 1, 0]).astype(np.float32) # 7
				self.class_2_attr[8] = np.array([1, 1, 1, 1, 1, 1, 1]).astype(np.float32) # 8
				self.class_2_attr[9] = np.array([1, 1, 1, 1, 0, 1, 0]).astype(np.float32) # 9
			else :
				print('[ERROR] 	Class number or attribute number does not match the default values of ' + str(CLASS_NUM_DEFAULT) + ' and ' + str(ATTR_NUM_DEFAULT) + ' respectively.')
				print('[ERROR] 	Terminating the program ...')
				sys.exit()
		# If not using default attribtues
		else :
			print('[INFO] 		Using custom attributes for class to attributes mapping. Please input the attributes as requested.')
			for a_class in range(self.class_num) :
				attr_list = input('[INFO] 		Please enter the attributes for class number ' + str(a_class) + ' in comma separated format [e.g., "1, 0, 1, ..." ] : \n[INFO]		')
				attrs = attr_list.strip().split(',')
				# If correct num of attrs, only then process
				if len(attrs) == self.attr_num :
					attrs_float = [float(x) for x in attrs]
					self.class_2_attr[a_class] = np.array(attrs_float).astype(np.float32)
					print('[INFO] 		Attribtues for class number ' + str(a_class) + ' updated successfully.')
				else :
					print('[ERROR] 		The number of attributes for class number ' + str(a_class) + ' does not match the attribute count of ' + str(self.attr_num) + '.')
					print('[ERROR] 		Terminating the program ...')
					sys.exit()

		# Show stats of the data loader
		print('[INFO] The class to attribute mapping used is as follows : ')
		for a_class in range(self.class_num) :
			print('[INFO] 		Class ' + str(a_class) + ' : ' + str(list(self.class_2_attr[a_class])))
		
		print('[INFO] Attributed MNIST data loader is initialized.')


	# Define a function to load next batch of inputs and the corresponding attributes
	def GetNextAttributedMNISTBatch(self, batch_size = BATCH_SIZE_DEFAULT, permissible_digits = '0,1,2,3,4,5,6,7,8,9', split = 'tr') :
		"""
		batch_size :
			The number of entries in a batch
		split :
			The data split of training, validation or testing ['tr', 'val', 'te']
		"""

		# Get list of permissible digits and a batch of theirs
		digits = [int(x.strip()) for x in permissible_digits.strip().split(',')]
		batch_digits = []
		for i in range(batch_size) :
			a_digit = random.choice(digits)
			batch_digits.append(a_digit)

		# Create the dataset [batch_size, 28*28]
		if split == 'tr' :
			indices = []
			for a_digit in batch_digits :
				indices.append(random.choice(self.train_digit_2_indices[a_digit]))
			x = self.mnist.train.images[indices]
			y = self.mnist.train.labels[indices]
		if split == 'val' :	
			indices = []
			for a_digit in batch_digits :
				indices.append(random.choice(self.validation_digit_2_indices[a_digit]))
			x = self.mnist.validation.images[indices]
			y = self.mnist.validation.labels[indices]
		if split == 'te' :
			indices = []
			for a_digit in batch_digits :
				indices.append(random.choice(self.test_digit_2_indices[a_digit]))
			x = self.mnist.test.images[indices]
			y = self.mnist.test.labels[indices]

		# Create attribute arrays
		all_attr = []
		attr_num = self.attr_num
		for an_attr in range(attr_num) :
			all_attr.append(np.zeros([batch_size, 1]))

		# Get attributes per class
		y_int = np.argmax(y, axis = 1)

		# Process each class attribtues
		for i in range(batch_size) :
			a_class = y_int[i]
			for an_attr in range(attr_num) :
				all_attr[an_attr][i] = self.class_2_attr[a_class, an_attr]

		# Return data, labels and attributes
		return x, y_int, all_attr


"""
# Main
if __name__ == "__main__" :
	# data_loader = dataLoaderForAttributedMNIST(class_num = 2, attr_num = 3, is_default_attr = False)
	data_loader = dataLoaderForAttributedMNIST(class_num = 10, attr_num = 7, is_default_attr = True)
	# data_loader = dataLoaderForAttributedMNIST(class_num = 2, attr_num = 3, is_default_attr = True)
	x_batch, y_batch, attr_batch = data_loader.GetNextAttributedMNISTBatch(batch_size = 3, split = 'tr')
	print(x_batch)
	print(list(y_batch))
	for i in range(3) :
		for j in range(7) :
			print(str(attr_batch[j][i]) + ' ',)
		print('\n')
	print(len(attr_batch))

	x_batch, y_batch, attr_batch = data_loader.GetNextAttributedMNISTBatch(batch_size = 3, split = 'val')
	print(x_batch)
	print(list(y_batch))
	for i in range(3) :
		for j in range(7) :
			print(str(attr_batch[j][i]) + ' ',)
		print('\n')
	print(len(attr_batch))

	x_batch, y_batch, attr_batch = data_loader.GetNextAttributedMNISTBatch(batch_size = 3, split = 'te')
	print(x_batch)
	print(list(y_batch))
	for i in range(3) :
		for j in range(7) :
			print(str(attr_batch[j][i]) + ' ',)
		print('\n')
	print(len(attr_batch))
"""