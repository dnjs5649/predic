import tensorflow as tf
import pandas as pd
import numpy as np
import pickle



data = pd.read_excel('201610날짜300r.xlsx')

test = pd.read_excel('201710날짜300r.xlsx')

data1 = pd.read_excel('201610an1soft.xlsx')

test1 = pd.read_excel('2017an1soft.xlsx')



with open('토큰1','wb') as mysavedata:
    pickle.dump(data, mysavedata)



with open('토큰2','wb') as mysavedata:
    pickle.dump(test, mysavedata)

with open('토큰3','wb') as mysavedata:
    pickle.dump(data1, mysavedata)


with open('토큰4','wb') as mysavedata:
    pickle.dump(test1, mysavedata)