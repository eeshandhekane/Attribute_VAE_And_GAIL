# Dependencies
import tensorflow as tf
import numpy as np
import os, sys, re
import Capsule_Variational_Auto_Encoder as cvae


# # Generate capsules and check on names
# cap1 = cvae.Capsule_Variational_Auto_Encoder(1, 2, 'cap1')
# print(cap1.cap_name)
# print(cap1.cap_name_list)
# cap1 = cvae.Capsule_Variational_Auto_Encoder(1, 2, 'cap2')
# print(cap1.cap_name)
# print(cap1.cap_name_list) # WORKS FINE


# # Generate capsules in a loop
# arr_caps = []
# for i in range(30):
# 	name = 'cap' + str(i)
# 	a_cap = cvae.Capsule_Variational_Auto_Encoder(name = name)
# 	arr_caps.append(a_cap)
# print(arr_caps)
# print(a_cap.cap_name)
# print(a_cap.cap_name_list) # WORKS FINE


# # Check passing of lists containing tf objects
# tf_entries_list = []
# for i in range(5):
# 	name = 'cap' + str(i)
# 	a_cap = cvae.Capsule_Variational_Auto_Encoder(X = '', KL_loss_list = [], attribute_loss_list = [], output_list = [], name = name, in_dim = 28*28, enc_dim = 100, latent_dim = 5, out_dim = 28*28)
# 	print(a_cap.cap_name)
# 	print(a_cap.cap_name_list) # WORKS FINE
# 	print(tf_entries_list) # WORKS FINE!!


# # Check if the lists are passed by reference or not
# l = []
# def foo(l, x) :
# 	l.append(x)
# for i in range(10) :
# 	foo(l, i)
# 	print(l)
# print(l)
# # [0]
# # [0, 1]
# # [0, 1, 2]
# # [0, 1, 2, 3]
# # [0, 1, 2, 3, 4]
# # [0, 1, 2, 3, 4, 5]
# # [0, 1, 2, 3, 4, 5, 6]
# # [0, 1, 2, 3, 4, 5, 6, 7]
# # [0, 1, 2, 3, 4, 5, 6, 7, 8]
# # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

net = cvae.capsuleVariationalAutoEncoder(10, 5)