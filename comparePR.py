import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('./results/thesisWhale6025664BoBFRPAKBiRelPrecRec.csv')

fRec = data['recs']
fPrec = data['precs']

data = pd.read_csv('./results/thesisWhale6025664FoBFRPAKBiRelPrecRec.csv')

bRec = data['recs']
bPrec = data['precs']






plt.figure(1)
plt.plot([0, 1], [1, 0], 'k--')
plt.plot(fRec,fPrec, color = 'blue', label = 'MC Blue')
plt.plot(bRec,bPrec, color = 'red', label = 'MC FIN')
plt.xlabel('Recall', fontsize = 26)
plt.ylabel('Precision', fontsize = 26)
plt.xticks(fontsize = 26) 
plt.yticks(fontsize = 26) 
plt.title('')
plt.legend(fontsize="26")
plt.show()