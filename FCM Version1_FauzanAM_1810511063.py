# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 13:36:09 2020

@author: DELL
"""

#Fauzan Akmal Mahdi
#1810511063

from __future__ import division, print_function
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skfuzzy as fuzz

#Jumlah cluster 	= 	c=2;
#Pangkat 	= 	w=2;
#Maksimum iterasi 	= 	MaxIter=100;
#Error terkecil yang diharapkan 	= 	.
#Fungsi obyektif awal 	= 	P0 = 0;
#Iterasi awal 	= 	t = 1;

colors = ['b', 'orange', 'g', 'r', 'c', 'm', 'y', 'k', 'Brown', 'ForestGreen']

data = 'Kuesioner Rating Kualitas Merk Laptop (Responses) - Form Responses 1.csv'
names = ['Timestamp','NamaResponden','InstansiResponden','MerkAsus','MerkLenovo','MerkAcer','MerkDell','MerkSamsung','MerkApple','UsiaResponden','JenisKelamin']

#dataframe
dataset = pd.read_csv(data, names=names)

#usia&jenis#data_clean_df = dataset.drop(columns=['Timestamp','NamaResponden','InstansiResponden','UsiaResponden','JenisKelamin'])
#dataset dalam bentuk dataframe
data_clean_df = dataset.drop(columns=['Timestamp','NamaResponden','InstansiResponden'])
data_clean_df = data_clean_df.drop([0])

#dataset dalam bentuk array
dataset_array = dataset.values
data_clean_arr = dataset_array[1:,3:9]

#Labeling
from sklearn import preprocessing
#membuat labelEncoder : mengubah variable string ke angka (sesuai urutan huruf)
le = preprocessing.LabelEncoder()

#pria:0, wanita:1
data_clean_df.JenisKelamin= le.fit_transform(data_clean_df.JenisKelamin)
#usia 18-25 tahun: 0, >25 tahun: 1, lain: 2
data_clean_df.UsiaResponden= le.fit_transform(data_clean_df.UsiaResponden)

#------------------------------------------------------------------------------
df1 = np.array(data_clean_df['MerkAsus']).astype(int)
#df2 = np.array(data_clean_df['JenisKelamin']).astype(int)
df2 = np.array(data_clean_df['MerkLenovo']).astype(int)

#------------------------------------------------------------------------------
# visualisasi data
fig0, ax0 = plt.subplots()
for label in range(2):
    ax0.plot(df1,df2, '.',color=colors[label])
#ax0.set_title('Test data: 200 points x3 clusters.')

plt.scatter(df1,df2,c=colors[label],alpha=0.5, edgecolor='k',cmap='viridis'),
plt.title("Plot Merk Asus dan Merk Lenovo")
plt.axis(xlim=(0,10), ylim=(0,10))

#-------------------------------------------------------------------------------------
# Menyiapkan data untuk looping dan data untuk ploting
fig1, axes1 = plt.subplots(3, 3, figsize=(8, 8))
alldata = np.vstack((df1, df2)).astype(int)
fpcs = []

#melakukan clustering untuk jumlah center 2 sampai 10
for ncenters, ax in enumerate(axes1.reshape(-1), 2):
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(alldata, ncenters, 2, error=0.005, maxiter=1000, init=None)
    
    #print(cntr,"\n\n", u, "\n\n",u0, "\n\n",d, "\n\n",jm, "\n\n",p, "\n\n",fpc)

    # Store fpc values for later
    fpcvalues = fpcs.append(fpc)
    fpcs.append(fpc)

    # Plot assigned clusters, for each data point in training set
    cluster_membership = np.argmax(u, axis=0)
    for j in range(ncenters):
        ax.plot(df1[cluster_membership == j],
                df2[cluster_membership == j], '.', color=colors[j])

    # Mark the center of each fuzzy cluster
    for pt in cntr:
        ax.plot(pt[0], pt[1], 'rs')

    ax.set_title('Centers = {0}; FPC = {1:.2f}'.format(ncenters, fpc))
    ax.axis('off')

fig1.tight_layout()
#------------------------------------------------------------------------------
