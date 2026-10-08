# Importing Standard Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Importing sklearn Librariesfrom ydata_profiling import ProfileReport
from sklearn import datasets
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sal = pd.read_csv("/home/arshadruk/Programs/Mini Projects/1/Salary.csv")

plt.style.use('ggplot')
sns.pairplot(sal)