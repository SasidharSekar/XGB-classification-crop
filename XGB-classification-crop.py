# %%
import warnings
import numpy
from pandas import DataFrame
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from scipy import stats
import xgboost as xgb
import pickle

# %% [markdown]
# Load the dataset

# %%
warnings.filterwarnings("ignore", category=UserWarning)
DATA_SET_URL = 'https://raw.githubusercontent.com/SasidharSekar/XGB-classification-crop/refs/heads/main/Crop_recommendation.csv'
data:DataFrame = read_csv(DATA_SET_URL,header=0,delimiter=',')

# %% [markdown]
# View Data Distribution

# %%
print("Data Size: %d" %data.size)
print(data.head(10))
print(data.describe())
print(data.groupby('label').size())
#excl_gender = data.iloc[:,1:]
print(data.iloc[:,:-1].corr())

# %% [markdown]
# Visualize Data Distribution

# %%
data.hist()
pyplot.show()
data.boxplot()
pyplot.show()
scatter_matrix(data)
pyplot.show()

# %% [markdown]
# Model Preparation

# %%
label_encoder = LabelEncoder()
data['label'] = label_encoder.fit_transform(data['label'])
array = data.values
X = array[:,:-1]
y = array[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33,random_state=42)

# %% [markdown]
# Model Evaluation

# %%
model = xgb.XGBClassifier(objective='binary:logistic',n_estimators = 100,learning_rate=0.1,max_depth=3,random_state=42)
kfold = StratifiedKFold(n_splits=10,shuffle=True,random_state=42)
cv_score = cross_val_score(model,X_train,y_train,cv=kfold,scoring='accuracy')
print('%f (%f)' %(cv_score.mean(),cv_score.std()))

# %% [markdown]
# Hyperparameter tuning

# %%
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1],
    'max_depth': [2, 3, 4],
}
grid_model = xgb.XGBClassifier(objective='binary:logistic')
grid_search = GridSearchCV(grid_model,param_grid=param_grid,cv=kfold, scoring='accuracy')
grid_search.fit(X_train,y_train)
param_dist = {
    'n_estimators': numpy.arange(100,500,100),
    'learning_rate': numpy.arange(0.01,0.2),
    'max_depth': numpy.arange(2,10,1),
    
    'subsample': numpy.arange(0.5,1,0.1),
    'lambda': numpy.arange(1,5,1)
}
random_model = xgb.XGBClassifier(objective='binary:logistic')
random_search = RandomizedSearchCV(random_model,param_distributions=param_dist,cv=kfold,scoring='accuracy')
random_search.fit(X_train,y_train)
search_space = {
    'n_estimators': Integer(100,300),
    'learning_rate': Real(0.01, 0.2),
    'max_depth': Integer(2,5),
    'subsample':
      Real(0.5, 1)
}
bayes_model = xgb.XGBClassifier(objective='binary:logistic')
bayes_search = BayesSearchCV(bayes_model,search_spaces=search_space,n_iter=20, cv=kfold,scoring='accuracy')
bayes_search.fit(X_train,y_train)

# %% [markdown]
# Hyperparameter tuning evaluation

# %%
best_grid_model = grid_search.best_estimator_
best_grid_params = grid_search.best_params_
grid_y_pred = best_grid_model.predict(X_test)
grid_accuracy = accuracy_score(y_test, grid_y_pred)
print('Best Grid parameters:', best_grid_params)
print('Test Grid accuracy:', grid_accuracy)
best_random_model = random_search.best_estimator_
best_random_params = random_search.best_params_
random_y_pred = best_random_model.predict(X_test)
random_accuracy = accuracy_score(y_test,random_y_pred)
print('Best Random parameters:', best_random_params)
print('Test Random accuracy:', random_accuracy)
best_bayes_params = bayes_search.best_params_
bayes_score = bayes_search.best_score_
print('Best Bayes parameters:', best_bayes_params)
print('Test Bayes accuracy:', bayes_score)

# %% [markdown]
# Make Predictions

# %%
model = xgb.XGBClassifier(**bayes_search.best_params_,objective='binary:logistic')
model.fit(X_train,y_train)
pred = model.predict(X_test)
print(accuracy_score(y_test,pred))
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))

# %% [markdown]
# Make individual prediction

# %%
str_x_indi = input('Enter input parameters as comma separated values')
x_indi = str_x_indi.split(',')
X_indi = numpy.array(x_indi)
X_indi = X_indi.reshape(1,-1)
X_indi = X_indi.astype(float)
pred = model.predict(X_indi)
pred = label_encoder.inverse_transform(pred)
print(pred)

# %% [markdown]
# Save Model to File

# %%
MODEL_FILE_NAME = 'final-model-xgb-crop.sav'
pickle.dump(model, open(MODEL_FILE_NAME,'wb'))

# %% [markdown]
# Load model from file and predict

# %%
model = pickle.load(open(MODEL_FILE_NAME,'rb'))
pred = model.predict(X_test)
print(accuracy_score(y_test,pred))


