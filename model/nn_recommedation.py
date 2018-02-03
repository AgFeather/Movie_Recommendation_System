#本项目使用卷积神经网络，使用MovieLens数据集完成电影推荐任务。
#数据集分为三个文件：用户数据，电影数据，评分数据
#用户数据：包含用户ID，性别，年龄，职业ID，邮编等字段
		   #UserID,Gender,Age,OccupationID,Zip-code
#电影数据：包含电影ID，电影名，电影风格等字段   
		   #MovieID, Title, Genres
#评分数据：包含用户ID，电影ID，评分，时间等字段
		   #UserID, MovieID, Rating, TimeStamp

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from collections import Counter
import tensorflow as tf

import os
import pickle
import re
from tensorflow.python.ops import math_ops

from urllib.request import urlretrieve
from os.path import isfile, isdir



features, target_values = pickle.load(open('features.p',mode='rb'))
title_count, title_set, genres2int, features, target_values,ratings, users,\
 movies, data, movies_orig, users_orig = pickle.load(open('params.p',mode='rb'))

#电影ID转下标的字典，数据集中电影ID跟下标不一致，比如第五行的数据电影ID不一定是5
movieid2idx = {val[0]: i for i,val in enumerate(movies.values)}
sentences_size = title_count #15
load_dir = './save_model/'






#获取 Tensors
def get_tensors(loaded_graph):
	uid = loaded_graph.get_tensor_by_name('uid:0')
	user_gender = loaded_graph.get_tensor_by_name('user_gender:0')
	user_age = loaded_graph.get_tensor_by_name('user_age:0')
	user_job = loaded_graph.get_tensor_by_name('user_job:0')
	movie_id = loaded_graph.get_tensor_by_name('movie_id:0')
	movie_categories = loaded_graph.get_tensor_by_name('movie_categories:0')
	movie_titles = loaded_graph.get_tensor_by_name('movie_titles:0')
	targets = loaded_graph.get_tensor_by_name('targets:0')
	dropout_keep_prob = loaded_graph.get_tensor_by_name('dropout_keep_prob:0')

	inference = loaded_graph.get_tensor_by_name('inference/MatMul:0')
	movie_combine_layer_flat = loaded_graph.get_tensor_by_name('movie_fc/Reshape:0')
	user_combine_layer_flat = loaded_graph.get_tensor_by_name('user_fc/Reshape:0')
	return uid, user_gender, user_age, user_job, movie_id, movie_categories,movie_titles, targets,\
		dropout_keep_prob, inference, movie_combine_layer_flat, user_combine_layer_flat


#预测指定用户对指定电影的评分
#这部分就是对网络做正向传播，计算得到预测的评分
def rating_movie(user_id_val, movie_id_val):
	loaded_graph = tf.Graph()
	with tf.Session(graph=loaded_graph) as sess:
		#load save model
		loader = tf.train.import_meta_graph(load_dir + '.meta')
		loader.restore(sess, load_dir)

		#get tensors from loaded model
		uid, user_gender, user_age, user_job, movie_id, movie_categories,movie_titles, targets,\
		dropout_keep_prob, inference,_,__ = get_tensors(loaded_graph)
		categories = np.zeros([1, 18])	
		categories[0] = movies.values[movieid2idx[movie_id_val]][2]
		titles = np.zeros([1, sentences_size])
		titles[0] = movies.values[movieid2idx[movie_id_val]][1]
		feed = {
			uid: np.reshape(users.values[user_id_val-1][0], [1, 1]),
			user_gender: np.reshape(users.values[user_id_val-1][1], [1, 1]),
			user_age: np.reshape(users.values[user_id_val-1][2], [1, 1]),
			user_job: np.reshape(users.values[user_id_val-1][3], [1, 1]),
			movie_id: np.reshape(movies.values[movieid2idx[movie_id_val]][0], [1, 1]),
			movie_categories: categories,  #x.take(6,1)
			movie_titles: titles,  #x.take(5,1)
			dropout_keep_prob: 1
		}

		#get prediction 
		inference_val = sess.run([inference], feed)
		return (inference_val)

#print('for user:234, predicting the rating for movie:1401', rating_movie(234, 1401))
#output[array([[4.279]], dtype=float32)]












#生成movie特征矩阵
#将训练好的电影特征组合成电影特征矩阵并保存到本地
#对每个电影进行正向传播
def save_movie_feature_matrix():
	pass
	loaded_graph = tf.Graph()
	movie_matrics = []
	with tf.Session(graph=loaded_graph) as sess:
		#load saved model
		loader = tf.train.import_meta_graph(load_dir + '.meta')
		loader.restore(sess, load_dir)

		#get tensor from loaded model
		uid, user_gender, user_age, user_job, movie_id, \
		movie_categories, movie_titles, targets, dropout_keep_prob,\
		_, movie_combine_layer_flat, __ = get_tensors(loaded_graph)

		for item in movies.values:
			categories = np.zeros([1, 18])
			categories[0] = item.take(2)

			titles = np.zeros([1, sentences_size])
			titles[0] = item.take(1)

			feed = {
				movie_id: np.reshape(item.take(0), [1, 1]),
				movie_categories:categories,#x.take(6,1)
				movie_titles:titles, #x.take(5, 1)
				dropout_keep_prob: 1,
			}

			movie_combine_layer_flat_val = sess.run([
				movie_combine_layer_flat], feed)
			movie_matrics.append(movie_combine_layer_flat_val)
	pickle.dump((
		np.array(movie_matrics).reshape(-1, 200)), open('movie_matrics.p','wb'))

#save_movie_feature_matrix()
#movie_matrics = pickle.load(open('movie_matrics.p',mode='rb'))









#生成user特征矩阵
#将训练好的用户特征组合成用户特征矩阵并保存到本地
#对每个用户进行正向传播
def save_user_feature_matrix():
	loaded_graph = tf.Graph()
	users_matrics = []
	with tf.Session(graph=loaded_graph) as sess:
		#load saved model
		loader = tf.train.import_meta_graph(load_dir + '.meta')
		loader.restore(sess, load_dir)
		uid, user_gender, user_age, user_job, movie_id, \
		movie_categories, movie_titles, targets, dropout_keep_prob,\
		_, __, user_combine_layer_flat = get_tensors(loaded_graph)
		for item in users.values:
			feed = {
				uid:np.reshape(item.take(0), [1, 1]),
				user_gender: np.reshape(item.take(1), [1, 1]),
				user_age: np.reshape(item.take(2), [1, 1]),
				user_job: np.reshape(item.take(3), [1, 1]),
				dropout_keep_prob: 1
			}
		user_combine_layer_flat_val = sess.run([user_combine_layer_flat], feed)
		users_matrics.append(user_combine_layer_flat_val)

	pickle.dump((
		np.array(users_matrics).reshape(-1, 200)), open('users_matrics.p', 'wb'))

#save_user_feature_matrix()
#users_matrics = pickle.load(open('users_matrics.p', mode='rb'))










#开始推荐电影
#是用产生的用户特征矩阵和电影特征矩阵做电影推荐

#推荐同类型的电影
#思路是计算指定电影的特征向量与整个电影特征矩阵的余弦相似度，
#取相似度最大的top_k个， 这里加了一些随机选择在里面，保证每次的推荐稍微不同
def recommend_same_type_movie(movie_id_val, top_k=20):
	loaded_graph = tf.Graph()
	movie_matrics = pickle.load(open('movie_matrics.p',mode='rb'))
	with tf.Session(graph=loaded_graph) as sess:
		#load saved model
		loader = tf.train.import_meta_graph(load_dir + '.meta')
		loader.restore(sess, load_dir)

		norm_movie_matrics = tf.sqrt(tf.reduce_sum(
			tf.square(movie_matrics), 1, keep_dims=True))
		normalized_movie_matrics = movie_matrics / norm_movie_matrics

		#推荐同类型的电影
		probs_embeddings = (movie_matrics[movieid2idx[movie_id_val]]).reshape([1, 200])
		probs_similarity = tf.matmul(probs_embeddings, tf.transpose(normalized_movie_matrics))
		sim = (probs_similarity.eval())

		print('the movies you may be interested in are:{}'.format(movies_orig[movieid2idx[movie_id_val]]))
		print('以下是给您的推荐：')
		p = np.squeeze(sim)
		p[np.argsort(p)[:-top_k]] = 0
		p = p / np.sum(p)
		results = set()
		while len(results) != 5:
			c = np.random.choice(3883, 1, p=p)[0]
			results.add(c)
		for val in (results):
			print(val)
			print(movies_orig[val])
		return results

#recommend_same_type_movie(1401, 20)














#推荐您喜欢的电影
#思路是使用用户特征向量与电影特征矩阵计算所有电影的评分，
#取评分最高的top_k个，同样加入随机选择
def recommend_your_favorite_movie(user_id_val, top_k=10):
	loaded_graph = tf.Graph()
	movie_matrics = pickle.load(open('movie_matrics.p',mode='rb'))
	users_matrics = pickle.load(open('users_matrics.p', mode='rb'))

	with tf.Session(graph=loaded_graph) as sess:
		#load saved model
		loader = tf.train.import_meta_graph(load_dir + '.meta')
		loader.restore(sess, load_dir)

		#推荐您喜欢的电影
		probs_embeddings = (users_matrics[user_id_val-1]).reshape([1, 200])

		probs_similarity = tf.matmul(probs_embeddings, tf.transpose(movie_matrics))
		sim = (probs_similarity.eval())

		print('以下是给您的推荐：')
		p = np.squeeze(sim)
		p[np.argsort(p)[:-top_k]] = 0
		p = p / np.sum(p)
		results = set()
		while len(results) != 5:
			c = np.random.choice(3883, 1, p=p)[0]
			results.add(c)
		for val in (results):
			print(val)
			print(movies_orig[val])
		return results

recommend_your_favorite_movie(234, 10)	














#看过这个电影的人还看了（喜欢）哪些电影
#首先选出喜欢某个电影的top_k个人，得到这几个人的用户特征向量
#然后计算这几个人对所有电影的评分
#选择每个人评分最高的电影作为推荐
#加入随机选择因素
import random

def recommend_other_favorite_movie(movie_id_val, top_k=20):
	loaded_graph = tf.Graph()
	movie_matrics = pickle.load(open('movie_matrics.p',mode='rb'))
	users_matrics = pickle.load(open('users_matrics.p', mode='rb'))
	with tf.Session(graph=loaded_graph) as sess:
		#load saved model 
		loader = tf.train.import_meta_graph(load_dir + '.meta')
		loader.restore(sess, load_dir)

		probs_movie_embeddings = (movie_matrics[movieid2idx[movie_id_val]]).reshape([1, 200])
		probs_user_favorite_similarity = tf.matmul(
			probs_movie_embeddings, tf.transpose(users_matrics))
		favorite_user_id = np.argsort(probs_user_favorite_similarity.eval())[0][-top_k:]
		# print(normalized_users_matrics.eval().shape)
		# print(probs_user_favorite_similarity.eval()[0][favorite_user_id])
		# print(favorite_user_id.shape)
		print('您看的电影是：{}'.format(movies_orig[movieid2idx[movie_id_val]]))
		print('喜欢看这个电影的人是：{}'.format(users_orig[favorite_user_id-1]))
		probs_user_embeddings = (users_matrics[favorite_user_id-1].reshape([-1, 200]))
		probs_similarity = tf.matmul(probs_user_embeddings, tf.transpose(movie_matrics))
		sim = (probs_similarity.eval())

		# results = (-sim[0]).argsort()[0:top_k]
		# print(results)
		# print(sim.shape)
		# print(np.argmax(sim, 1))
		p = np.argmax(sim, 1)
		print('喜欢这个电影的人还喜欢看：')

		results = set()
		while len(results) != 5:
			c = p[random.randrange(top_k)]
			results.add(c)


		for val in (results):
			print(val)
			print(movies_orig[val])
		return results

recommend_other_favorite_movie(1401, 20)


























# 加载数据并保存到本地
# title_count：Title字段的长度（15）
# title_set：Title文本的集合
# genres2int：电影类型转数字的字典
# features：是输入X
# targets_values：是学习目标y
# ratings：评分数据集的Pandas对象
# users：用户数据集的Pandas对象
# movies：电影数据的Pandas对象
# data：三个数据集组合在一起的Pandas对象
# movies_orig：没有做数据处理的原始电影数据
# users_orig：没有做数据处理的原始用户数据