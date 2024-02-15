import pandas as pd
import pickle


infile = open('./BallenyIslands2015/BallenyIslands2015DF60FBWhale.pkl','rb')
data = pickle.load(infile)
infile.close()

data['dataset'] = 'BallenyIslands2015'

infile = open('./casey2014/casey2014DF60FBWhale.pkl','rb')
data2 = pickle.load(infile)
infile.close()

data2['dataset'] = 'casey2014'

data = pd.concat([data,data2])

infile = open('./casey2017/casey2017DF60FBWhale.pkl','rb')
data2 = pickle.load(infile)
infile.close()

data2['dataset'] = 'casey2017'

data = pd.concat([data,data2])

infile = open('./ElephantIsland2013Aural/ElephantIsland2013AuralDF60FBWhale.pkl','rb')
data2 = pickle.load(infile)
infile.close()

data2['dataset'] = 'ElephantIsland2013Aural'

data = pd.concat([data,data2])

infile = open('./ElephantIsland2014/ElephantIsland2014DF60FBWhale.pkl','rb')
data2 = pickle.load(infile)
infile.close()

data2['dataset'] = 'ElephantIsland2014'

data = pd.concat([data,data2])

infile = open('./Greenwich64S2015/Greenwich64S2015DF60FBWhale.pkl','rb')
data2 = pickle.load(infile)
infile.close()

data2['dataset'] = 'Greenwich64S2015'

data = pd.concat([data,data2])

infile = open('./kerguelen2005/kerguelen2005DF60FBWhale.pkl','rb')
data2 = pickle.load(infile)
infile.close()

data2['dataset'] = 'kerguelen2005'

data = pd.concat([data,data2])

infile = open('./kerguelen2014/kerguelen2014DF60FBWhale.pkl','rb')
data2 = pickle.load(infile)
infile.close()

data2['dataset'] = 'kerguelen2014'

data = pd.concat([data,data2])

infile = open('./kerguelen2015/kerguelen2015DF60FBWhale.pkl','rb')
data2 = pickle.load(infile)
infile.close()

data2['dataset'] = 'kerguelen2015'

data = pd.concat([data,data2])

infile = open('./MaudRise2014/MaudRise2014DF60FBWhale.pkl','rb')
data2 = pickle.load(infile)
infile.close()

data2['dataset'] = 'MaudRise2014'

data = pd.concat([data,data2])

infile = open('./RossSea2014/RossSea2014DF60FBWhale.pkl','rb')
data2 = pickle.load(infile)
infile.close()

data2['dataset'] = 'RossSea2014'

data = pd.concat([data,data2])

pickle.dump(data, open('./Dataframes/allDatasets250_60FBWhale.pkl', 'wb'))