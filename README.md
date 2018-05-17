# 电影推荐系统
## 关于本项目
本项目是使用python3 + tensorflow，搭建一个基于卷积神经网络模型的离线电影推荐系统，电影数据集使用的是MovieLens中的ml-1m.zip数据集。

## 技术说明
- python：3.5.2
- tensorflow: 1.4.0
- numpy: 1.14.0

## 如何运行
该项目目前还没有对整个流程进行集成，所以需要按照步骤顺序运行对应文件
1. 运行data_download.py 文件，下载对应的数据集并进行校验
2. 运行data_processing.py 文件，对下载的数据集进行处理，并将处理产生的特征存储在model文件夹中生成features.p 和 params.p 文件
3. 运行model文件夹中的training.py文件，对神经网络进行训练
4. 运行model文件夹中的recommendation文件，测试各种推荐方法


## 关于数据集
本项目使用的是MovieLens的ml-1m数据集，该数据集合包含6,040名用户对3,900个电影的1,000,209个匿名评论。

数据集包括movies.dat, ratings.dat, users.dat三个文件

#### movies.dat
该数据集存储了电影信息，包含字段：MovieID，Title，Genres
- MovieID: 电影ID(1-3952)
- Title：电影标题，包括出版年份
- Genres：电影类别（包括喜剧，动作剧，纪录片等..)

详细内容可以参照ml-1m/README

#### users.dat
该数据集包含了对电影进行评分的用户信息，包括字段：UserID，Gender，Age，Occupation，Zip-code
- UserID：用户ID(1-6040)
- Gender：性别（“M” or “F”）
- Age：年龄，该年龄不是连续变量，而是被分为7个年龄集合（under 18；18-24；25-34；35-44...）
- Occupation：职业，这里用数字0-20表示各个职业

详细内容可以参照ml-1m/README

#### ratings.dat
该数据集是用户对电影的评分，包括字段：UserID::MovieID::Rating::Timestamp。  
其中rating取值为：0，1，2，3，4，5  
Timestamp表示时间戳  
每个用户有最少20个评分

详细内容可以参照ml-1m/README
