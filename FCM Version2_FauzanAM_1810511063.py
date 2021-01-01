# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 19:54:15 2020

@author: recov
"""

#Fauzan Akmal Mahdi
#1810511063

from __future__ import division, print_function
import numpy as np
import pandas as pd

colors = ['b', 'orange', 'g', 'r', 'c', 'm', 'y', 'k', 'Brown', 'ForestGreen']

data = 'Kuesioner Rating Kualitas Merk Laptop (Responses) - Form Responses 1.csv'
names = ['Timestamp','NamaResponden','InstansiResponden','MerkAsus','MerkLenovo','MerkAcer','MerkDell','MerkSamsung','MerkApple','UsiaResponden','JenisKelamin']

#dataframe
dataset = pd.read_csv(data, names=names)

#usia&jenis#data_clean_df = dataset.drop(columns=['Timestamp','NamaResponden','InstansiResponden','UsiaResponden','JenisKelamin'])
data_clean_df = dataset.drop(columns=['Timestamp','NamaResponden','InstansiResponden'])
data_clean_df = data_clean_df.drop([0])

#array
dataset_array = dataset.values
data_clean_arr = dataset_array[1:,3:9]

#Labeling
from sklearn import preprocessing
#membuat labelEncoder : mengubah variable string ke angka (sesuai urutan huruf)
le = preprocessing.LabelEncoder()

#labeling
#pria:0, wanita:1
data_clean_df.JenisKelamin= le.fit_transform(data_clean_df.JenisKelamin)
#usia 18-25 tahun: 0, >25 tahun: 1, lain: 2
data_clean_df.UsiaResponden= le.fit_transform(data_clean_df.UsiaResponden)

#------------------------------------------------------------------------------
#dua kelas yang ingin dilakukan clustering
df1 = np.array(data_clean_df['MerkAsus']).astype(int)
df2 = np.array(data_clean_df['MerkLenovo']).astype(int)

#impor library yang dibutuhkan
from fcmeans import FCM
from matplotlib import pyplot as plt
from seaborn import scatterplot as scatter

#centers = [(0, 0), (5, 5)]

data_array = np.array(data_clean_df).astype(int)

# fit the fuzzy-c-means
fcm = FCM(n_clusters=2)
fcm.fit(data_array)

# outputs
fcm_centers = fcm.centers
fcm_labels = fcm.u.argmax(axis=1)

#dalam bentuk array
center = np.array(fcm_centers)

#plot untuk menampilkan result
f, axes = plt.subplots(1,2, figsize=(11,5))
scatter(df1, df2, ax=axes[0])
scatter(df1, df2, ax=axes[0], hue=fcm_labels)
scatter(center[:,0], center[:,1], ax=axes[1], marker="s",s=200)
plt.title('Plot Merk Asus dan Merk Lenovo')
plt.show()
