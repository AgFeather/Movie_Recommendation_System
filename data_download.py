import os
from urllib.request import urlretrieve
from os.path import isfile, isdir
from tqdm import tqdm
import zipfile
import hashlib





def download_data():
	"""
	download movie data
	"""
	data_name = 'ml-1m'
	save_path = './ml-1m.zip'
	url = 'http://files.grouplens.org/datasets/movielens/ml-1m.zip'
	hash_code = 'c4d9eecfca2ab87c1945afe126590906'
	if os.path.exists(save_path):
		if hashlib.md5(open(save_path, 'rb').read()).hexdigest() == hash_code:
			print('found and verify {} data'.format(data_name))
			return
		else:
			print('{} file is corrupted, remove it and try again'.format(save_path))# remove and try again automatically
			return
	else:
		with DLProgress(unit='B', unit_scale=True, miniters=1, desc='Downloading {}'.format(data_name)) as pbar:
			urlretrieve(url, save_path, pbar.hook)


def extract_data():
	"""
	extract data from ml-1m.zip
	"""
	data_name = 'ml-1m'
	data_path = './ml-1m.zip'
	extract_path = './'
	extract_fn = unzip
	if not os.path.exists(extract_path):
		os.makedirs(extract_path)
	try:
		extract_fn(data_name, data_path, extract_path)
	except Exception as err:
		shutil.rmtree(extract_path) #remove extraction folder if there is an error
		raise err
	print('extracting done')

def unzip(data_name, from_path, to_path):
	print('extracting {} ....'.format(data_name))
	with zipfile.ZipFile(from_path) as zf:
		zf.extractall(to_path)



class DLProgress(tqdm):
	"""
	Handle progress bar while downloading
	"""
	last_block = 0
	def hook(self, block_num=1, block_size=1, total_size=None):
		"""
		a hook function that will be called once on establishment of the 
		network connection and once after each block read thereafter
		parameter:
		block_num:a count of blocks transferred so far
		block_size: block size in bytes
		total_size: the total size of the file. this may be -1 on older
		FTP servers with do not return a file size in response to retrieval reqeust
		"""
		self.total=total_size
		self.update((block_num - self.last_block) * block_size)
		self.last_block = block_num



if __name__ == '__main__':
	download_data()
	extract_data()