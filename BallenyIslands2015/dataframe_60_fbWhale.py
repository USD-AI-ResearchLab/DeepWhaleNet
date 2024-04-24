
import numpy as np
import pandas as pd
import pickle
import os
from pathlib import Path
import librosa

# given a folder full of wav files\n",

ss = 16384
hop = 8192

p = Path(r'./wav').glob('**/*')
files = [x for x in p if x.is_file()]

fileName = []
chunks = []
array = []
label = []


for file in files:
    if ('.wav') in str(file):
        None
    else:
        continue
    y, sr = librosa.load(file, sr = 250)
    for i in range(int(len(y)/hop)):
        z = np.zeros(ss)
        z = y[int(i*hop):int((i*hop)+ss)]
        fileName.append(f'{file}')
        chunks.append(i)
        array.append(z)
        label.append(np.zeros(2))

def labeler(data,chunk,ss,hop,label):
    for row in data.itertuples():
        for chunk in pddf.itertuples():
            if (('wav\\' + str(row[4]))!=('wav\\' + str(row[5])) ):
                if (('wav\\' + str(row[4])) == str(chunk.fileName) and chunk.chunks*hop*4 <= row[8] and ((chunk.chunks*hop)+ss)*4 >= row[8]) or (('wav\\' + str(row[5])) == str(chunk.fileName) and ((chunk.chunks*hop)+ss)*4 >= row[9] and chunk.chunks*hop*4 <= row[9]):
                    chunk.label[label] = 1
                continue
            else:
                if ('wav\\' + str(row[4])) == str(chunk.fileName) and chunk.chunks*hop*4 <= row[8] and ((chunk.chunks*hop)+ss)*4 >= row[9]:
                    chunk.label[label] = 1
                elif ('wav\\' + str(row[4])) == str(chunk.fileName) and chunk.chunks*hop*4 <= row[8] and ((chunk.chunks*hop)+ss)*4 >= row[8] and ((chunk.chunks*hop)+ss)*4 <= row[9] or ('wav\\' + str(row[4])) == str(chunk.fileName) and chunk.chunks*hop*4 >= row[8] and (chunk.chunks*hop)*4 <= row[9] and ((chunk.chunks*hop)+ss)*4 >= row[9]:
                    #remove.append(row[0])
                    chunk.label[label] = 1    

# return array of file name, with the chunk number, the audio chunk  \n",
#   #"

pddf = pd.DataFrame(data={'fileName':fileName,'chunks':chunks, 'array':array, 'label':label})
print(pddf)


# given the chunk array\n",
# 
data = pd.read_csv('./Balleny2015.BmAnt-A.selections.txt', sep = '\t')
labeler(data,pddf,ss,hop,0)


data = pd.read_csv('./Balleny2015.BmAnt-B.selections.txt', sep = '\t')
labeler(data,pddf,ss,hop,0)

data = pd.read_csv('./Balleny2015.BmAnt-Z.selections.txt', sep = '\t')
labeler(data,pddf,ss,hop,0)

data = pd.read_csv('./Balleny2015.BmD.selections.txt', sep = '\t')
labeler(data,pddf,ss,hop,0)

data = pd.read_csv('./Balleny2015.Bp20Hz.selections.txt', sep = '\t')
labeler(data,pddf,ss,hop,1)

data = pd.read_csv('./Balleny2015.Bp20Plus.selections.txt', sep = '\t')
labeler(data,pddf,ss,hop,1)

data = pd.read_csv('./Balleny2015.BpDwnswp.selections.txt', sep = '\t')
labeler(data,pddf,ss,hop,1)

pd.set_option('display.max_rows', pddf.shape[0]+1)
print(pddf.label)
  
pickle.dump(pddf, open('./BallenyIslands2015DF60FBWhale.pkl', 'wb'))
