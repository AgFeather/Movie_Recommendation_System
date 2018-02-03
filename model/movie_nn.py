import tensorflow as tf
import pickle


features, target_values = pickle.load(open('features.p',mode='rb'))
title_count, title_set, genres2int, features, target_values,ratings, users,\
 movies, data, movies_orig, users_orig = pickle.load(open('params.p',mode='rb'))



MOVIE_CATEGORAY_LENGTH = 18
MOVIE_TITLE_LENGTH = 15

embed_dim = 32
#电影ID个数
movie_id_max = max(features.take(1,1)) + 1#3952
#电影类型个数
movies_categories_max = max(genres2int.values()) + 1#18+1
#电影名单词个数
movies_title_max = len(title_set) #5216

#对电影类型嵌入向量做加和操作的标志，考虑过使用mean做平均，但是没有实现mean
combiner = 'sum'

#电影名长度
sentences_size = title_count #15
#文本卷积滑动窗口，分别滑动2,3,4,5个单词
window_sizes = [2,3,4,5]
#文本卷积数量
filter_num = 8

#电影ID转下标的字典，数据集中电影ID跟下标不一致，比如第五行的数据电影ID不一定是5
movieid2idx = {val[0]: i for i,val in enumerate(movies.values)}


#超参
num_epochs = 5
batch_size = 256

dropout_keep = 0.5
learning_rate = 0.0001
#show stats for every n number of batches
show_every_n_batches = 20

save_dir = './save'





def get_inputs():
	movie_id = tf.placeholder(tf.int32, [None,1], name='movie_id')
	movie_categories = tf.placeholder(tf.int32, [None, MOVIE_CATEGORAY_LENGTH], name='movie_categories')
	movie_titles = tf.placeholder(tf.int32, [None,MOVIE_TITLE_LENGTH], name='movie_titles')
	dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
	return movie_id, movie_categories, movie_titles, dropout_keep_prob

#定义Movie ID的嵌入矩阵
def get_movie_id_embed_layer(movie_id):
	with tf.name_scope('movie_embedding'):
		movie_id_embed_matrix = tf.Variable(tf.random_uniform([
			movie_id_max, embed_dim], -1, 1), name='movie_id_embed_matrix')
		movie_id_embed_layer = tf.nn.embedding_lookup(
			movie_id_embed_matrix, movie_id, name='movie_id_embed_layer')
	return movie_id_embed_layer

#对电影类型的多个嵌入向量做加和
def get_movie_categories_embed_layer(movie_categories):
	with tf.name_scope('movie_categories_layer'):
		movie_categories_embed_matrix = tf.Variable(tf.random_uniform([
			movies_categories_max, embed_dim], -1, 1), 
			name='movie_categories_embed_matrix')
		movie_categories_embed_layer = tf.nn.embedding_lookup(
			movie_categories_embed_matrix, movie_categories,
			name='movie_categories_embed_layer')
		if combiner == 'sum':
			movie_categories_embed_layer = tf.reduce_sum(
				movie_categories_embed_layer, axis=1, keep_dims=True)
	return movie_categories_embed_layer

#movie title 的文本卷积网络实现
def get_movie_cnn_layer(movie_titles, dropout_keep_prob):
	#从嵌入矩阵中得到电影名对应的各个单词的嵌入向量
	with tf.name_scope('movie_embedding'):
		movie_title_embed_matrix = tf.Variable(tf.random_uniform([
			movies_title_max, embed_dim], -1, 1), 
			name='movie_title_embed_matrix')
		movie_title_embed_layer = tf.nn.embedding_lookup(
			movie_title_embed_matrix, movie_titles,
			name='movie_title_embed_layer')
		movie_title_embed_layer_expand = tf.expand_dims(movie_title_embed_layer, -1)#??no understand

	#对文本嵌入层使用不同尺寸的卷积核做卷积核最大池化
	pool_layer_lst = []
	for window_size in window_sizes:
		with tf.name_scope('movie_txt_conv_maxpool_{}'.format(window_size)):
			filter_weights = tf.Variable(tf.truncated_normal([
				window_size, embed_dim, 1, filter_num], stddev=0.1), name='filter_weights')#molify the size of kernel
			filter_bias = tf.Variable(tf.constant(0.1, shape=[filter_num], name='filter_bias'))
			conv_layer = tf.nn.conv2d(movie_title_embed_layer_expand,
				filter_weights, [1,1,1,1], padding='VALID', name='conv_layer')
			relu_layer = tf.nn.relu(tf.nn.bias_add(conv_layer, filter_bias), name='relu_layer')
			maxpool_layer = tf.nn.max_pool(relu_layer, 
				[1, sentences_size - window_size + 1, 1, 1],
				[1,1,1,1], padding='VALID', name='maxpool_layer')
			pool_layer_lst.append(maxpool_layer)

	#dropout layer
	with tf.name_scope('pool_dropout'):
		pool_layer = tf.concat(pool_layer_lst, 3, name='pool_layer')
		max_num = len(window_sizes) * filter_num
		pool_layer_flat = tf.reshape(pool_layer, [-1, 1, max_num], name='pool_layer_flat')
		dropout_layer = tf.nn.dropout(pool_layer_flat, dropout_keep_prob, name='dropout_layer')
	return pool_layer_flat, dropout_layer

#将movie的各个层一起做全连接
def get_movie_feature_layer(
	movie_id_embed_layer, movie_categories_embed_layer, dropout_layer):
	with tf.name_scope('movie_fc'):
		# first full connection layer
		movie_id_fc_layer = tf.layers.dense(movie_id_embed_layer,
			embed_dim, name='movie_id_fc_layer', activation=tf.nn.relu)
		movie_categories_fc_layer = tf.layers.dense(movie_categories_embed_layer,
			embed_dim, name='movie_categories_fc_layer', activation=tf.nn.relu)

		#second full connection layer
		movie_combine_layer = tf.concat([
			movie_id_fc_layer, movie_categories_fc_layer, dropout_layer], 2)#(?,1,96)
		movie_combine_layer = tf.contrib.layers.fully_connected(
			movie_combine_layer, 200, tf.tanh) #(?,1,200)
		movie_combine_layer_flat = tf.reshape(movie_combine_layer, [-1, 200])
	return movie_combine_layer, movie_combine_layer_flat







if __name__ == '__main__':
	movie_id, movie_categories, movie_titles, dropout_keep_prob = get_inputs()
	movie_id_embed_layer = get_movie_id_embed_layer(movie_id)
	movie_categories_embed_layer = get_movie_categories_embed_layer(movie_categories)
	pool_layer_flat, dropout_layer = get_movie_cnn_layer(movie_titles, dropout_keep_prob)
	movie_combine_layer, movie_combine_layer_flat = get_movie_feature_layer(
		movie_id_embed_layer, movie_categories_embed_layer, dropout_layer)