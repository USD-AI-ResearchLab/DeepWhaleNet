import subprocess
import os


os.chdir('./BallenyIslands2015')
subprocess.run(['py', './dataframe_60_fbWhale.py'])
os.chdir('..')

os.chdir('./casey2014')
subprocess.run(['py', './dataframe2_60_fbWhale.py'])
os.chdir('..')

os.chdir('./casey2017')
subprocess.run(['py', './dataframe3_60_fbWhale.py'])
os.chdir('..')

os.chdir('./ElephantIsland2013Aural')
subprocess.run(['py', './dataframe4_60_fbWhale.py'])
os.chdir('..')

os.chdir('./ElephantIsland2014')
subprocess.run(['py', './dataframe5_60_fbWhale.py'])
os.chdir('..')

os.chdir('./Greenwich64S2015')
subprocess.run(['py', './dataframe6_60_fbWhale.py'])
os.chdir('..')

os.chdir('./kerguelen2005')
subprocess.run(['py', './dataframe7_60_fbWhale.py'])
os.chdir('..')

os.chdir('./kerguelen2014')
subprocess.run(['py', './dataframe8_60_fbWhale.py'])
os.chdir('..')

os.chdir('./kerguelen2015')
subprocess.run(['py', './dataframe9_60_fbWhale.py'])
os.chdir('..')

os.chdir('./MaudRise2014')
subprocess.run(['py', './dataframe10_60_fbWhale.py'])
os.chdir('..')

os.chdir('./RossSea2014')
subprocess.run(['py', './dataframe11_60_fbWhale.py'])
os.chdir('..')

subprocess.run(['py', './concatDataframes.py'])

subprocess.run(['py', './thesisWhaleCross60SFB.py'])

subprocess.run(['py', './thesisWhaleTest60SBF.py'])

subprocess.run(['py', './thesisWhaleTestRPAK60SBF.py'])

subprocess.run(['py', './comparePR.py'])