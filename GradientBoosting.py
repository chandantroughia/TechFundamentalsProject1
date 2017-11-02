import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

