import tensorflow as tf
import pickle


features, target_values = pickle.load(open('features.p',mode='rb'))
title_count, title_set, genres2int, features, target_values,ratings, users,\
 movies, data, movies_orig, users_orig = pickle.load(open('params.p',mode='rb'))



MOVIE_CATEGORAY_LENGTH = 18
MOVIE_TITLE_LENGTH = 15
#嵌入矩阵的维度
embed_dim = 32
#用户ID个数
uid_max = max(features.take(0,1)) + 1#6040
#性别个数
gender_max = max(features.take(2,1)) + 1 #1+1 = 2
#年龄类别个数
age_max = max(features.take(3,1)) + 1 #6+1 = 7
#职业个数
job_max = max(features.take(4,1)) + 1 #20 + 1 = 21






def get_inputs():
	uid = tf.placeholder(tf.int32, [None,1], name='uid')
	user_gender = tf.placeholder(tf.int32, [None,1], name='user_gender')
	user_age = tf.placeholder(tf.int32, [None,1], name='user_age')
	user_job = tf.placeholder(tf.int32, [None,1], name='user_job')
	return uid, user_gender, user_age, user_job


def get_user_embedding(uid, user_gender, user_age, user_job):
	with tf.name_scope('user_embedding'):
		uid_embed_metrix = tf.Variable(tf.random_uniform([
			uid_max, embed_dim], -1, 1), name='uid_embed_metrix')
		uid_embed_layer = tf.nn.embedding_lookup(
			uid_embed_metrix, uid, name='uid_embed_layer')
		gender_embed_matrix = tf.Variable(tf.random_uniform([
			gender_max, embed_dim // 2], -1, 1), name='gender_embed_matrix')#donot need gender embedding
		gender_embed_layer = tf.nn.embedding_lookup(
			gender_embed_matrix, user_gender, name='gender_embed_layer')
		age_embed_matrix = tf.Variable(tf.random_uniform([
			age_max, embed_dim // 2], -1, 1), name='age_embed_matrix')
		age_embed_layer = tf.nn.embedding_lookup(
			age_embed_matrix, user_age, name='age_embed_layer')
		job_embed_matrix = tf.Variable(tf.random_uniform([
			job_max, embed_dim // 2], -1, 1), name='job_embed_matrix')
		job_embed_layer = tf.nn.embedding_lookup(
			job_embed_matrix, user_job, name='job_embed_layer')
	return uid_embed_layer, gender_embed_layer, age_embed_layer, job_embed_layer

#将user的嵌入矩阵一起全连接生成user的特征
def get_user_feature_layer(
	uid_embed_layer, gender_embed_layer, age_embed_layer, job_embed_layer):
	with tf.name_scope('user_fc'):
		#first full connection layer
		uid_fc_layer = tf.layers.dense(
			uid_embed_layer, embed_dim, name='uid_fc_layer', activation=tf.nn.relu)
		gender_fc_layer = tf.layers.dense(
			gender_embed_layer, embed_dim, name='gender_fc_layer', activation=tf.nn.relu)
		age_fc_layer = tf.layers.dense(
			age_embed_layer, embed_dim, name='age_fc_layer', activation=tf.nn.relu)
		job_fc_layer = tf.layers.dense(
			job_embed_layer, embed_dim, name='job_fc_layer', activation=tf.nn.relu)

		#second full connection layer
		user_combine_layer = tf.concat([
			uid_fc_layer, gender_fc_layer,age_fc_layer, job_fc_layer], 2)#(?,1,128)
		user_combine_layer = tf.contrib.layers.fully_connected(
			user_combine_layer, 200, tf.tanh) #(?, 1, 200) -->(?, 128, 200)
		user_combine_layer_flat = tf.reshape(user_combine_layer, [-1, 200])
	return user_combine_layer, user_combine_layer_flat





if __name__ == '__main__':
	uid, user_gender, user_age, user_job = get_inputs()
	uid_embed_layer, gender_embed_layer, age_embed_layer, job_embed_layer = get_user_embedding(
		uid, user_gender, user_age, user_job)
	user_combine_layer, user_combine_layer_flat = get_user_feature_layer(
		uid_embed_layer, gender_embed_layer, age_embed_layer, job_embed_layer)