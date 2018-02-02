import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import re
import pickle


title_count = 15


"""
数据预处理
UserID、Occupation和MovieID不用变。


"""
def user_data_processing():
	'''
	Gender字段：需要将‘F’和‘M’转换成0和1。
	Age字段：要转成7个连续数字0~6。
	discard zip-code
	'''
	#read user data
	print('user_data_processing....')
	user_title = ['UserID','Gender','Age','JobID','Zip-code']
	users = pd.read_table('./ml-1m/users.dat', sep='::', header=None,
		names=user_title, engine='python')#pandas 读取数据方法？
	users = users.filter(regex='UserID|Gender|Age|JobID')#？filter用法
	users_orig = users.values #？返回所有对象 numpy.ndarray
	#mollify 'users_orig'

	#print(users_orig[0:10])

	#transfer gender and age
	gender_map = {'F':0,'M':1}
	users['Gender'] = users['Gender'].map(gender_map)#学习整理
	age_map = {val:ii for ii, val in enumerate(set(users['Age']))}
	users['Age'] = users['Age'].map(age_map)#学习整理

	return users, users_orig




def movie_data_processing():
	'''
	Genres字段：是分类字段，转成数字。
	首先将Genres中的类别转成字符串到数字的字典，
	因为有些电影是多个Genres的组合,xuyao再将每个电影的Genres字段转成数字列表.
	Title字段：处理方式跟Genres字段一样，首先创建文本到数字的字典，
	然后将Title中的描述转成数字的列表。另外Title中的年份也需要去掉。
	Genres和Title字段需要将长度统一，这样在神经网络中方便处理。
	空白部分用‘< PAD >’对应的数字填充。
	'''
	#read movie dataset
	print('movie_data_processing....')
	movies_title = ['MovieID', 'Title', 'Genres']
	movies = pd.read_table('./ml-1m/movies.dat', sep='::',
		header=None, names=movies_title, engine='python')
	movies_orig = movies.values#length:3883
#	print(movies_orig[0:10])

	# remove years from movie title
	pattern = re.compile(r'^(.*)\((\d+)\)$')#学习
	title_map = {val:pattern.match(val).group(1) for ii, val in enumerate(set(movies['Title']))}
	movies['Title'] = movies['Title'].map(title_map)

	#电影类型转为数字字典
	genres_set = set()
	for val in movies['Genres'].str.split('|'):#？？学习
		genres_set.update(val)#update()添加多项  add()添加1项
	genres_set.add('<PAD>')
	genres2int = {val:ii for ii, val in enumerate(genres_set)} # length:19

	#for each movie, transfer the its genres to int list with same length:18
	genres_map={val:[genres2int[row] for row in val.split('|')]\
			for val in set(movies['Genres'])}
			#for ii val in enumerate(set(movies['Genres']))}
	for key in genres_map:#padding with the same length operation
		for cnt in range(max(genres2int.values()) - len(genres_map[key])):
			genres_map[key].insert(len(genres_map[key]) + cnt, genres2int['<PAD>'])#

	movies['Genres'] = movies['Genres'].map(genres_map)

	#transfer movie's title to digital dir
	title_set = set()
	for val in movies['Title'].str.split():
		title_set.update(val)

	title_set.add('<PAD>')
	title2int = {val:ii for ii, val in enumerate(title_set)}#length:5215

	#transfer movie's title to int list with same length:15

	title_map = {val:[title2int[row] for row in val.split()] \
					for val in set(movies['Title'])}
			#for ii, val in enumerate(set(movies['Title']))}
	for key in title_map:#padding with the same length operation
		for cnt in range(title_count - len(title_map[key])):
			title_map[key].insert(len(title_map[key]) + cnt, title2int['<PAD>'])

	movies['Title'] = movies['Title'].map(title_map)

	return movies, movies_orig, genres2int,title_set


def rating_data_processing():
	#read ratings dataset
	print('rating_data_processing....')
	ratings_title = ['UserID', 'MovieID', 'ratings', 'timestamps']
	ratings = pd.read_table('./ml-1m/ratings.dat', sep='::',
		header=None, names=ratings_title, engine='python')
	ratings = ratings.filter(regex='UserID|MovieID|ratings')

	return ratings



def get_feature():
	"""
	load dataset from file
	data processing
	"""
	users, users_orig = user_data_processing()
	movies, movies_orig, genres2int,title_set = movie_data_processing()
	ratings = rating_data_processing()

	#merge three tables
	data = pd.merge(pd.merge(ratings, users), movies)

	#split data to feature set:X and lable set:y
	target_fields = ['ratings']
	feature_pd, tragets_pd = data.drop(target_fields, axis=1), data[target_fields]
	features = feature_pd.values
	target_values = tragets_pd.values

	print(type(feature_pd))
	print(feature_pd.head())

	save_params((title_count, title_set, genres2int, features, target_values,\
	 		ratings, users, movies, data, movies_orig, users_orig))

	pickle.dump((features, target_values),open('./model/features.p','wb'))

	return features, target_values


def save_params(params):
	print('saving parameters as params.p ....')
	pickle.dump(params, open('./model/params.p', 'wb'))



if __name__ == '__main__':
	get_feature()
