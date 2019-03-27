# Load CSV using Pandas from URL
from pandas import read_csv
# this data set does not exist anymore : https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes
url = "https://goo.gl/vhm1eU"  
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
# but there are many others that exist at this address!
data = read_csv(url, names=names)
print(data.shape)