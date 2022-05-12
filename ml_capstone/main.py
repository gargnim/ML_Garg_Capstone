import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
from matplotlib import pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LinearRegression, LogisticRegression
import statsmodels.api as sm
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn import svm
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
import sklearn.metrics as metrics
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from scipy import stats
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import f1_score

np.random.seed(2912)

# spy = yf.download('SPY','2001-12-31','2021-12-31')
# spy.to_csv('spy_data.csv')
spy = pd.read_csv('spy_data.csv')
pe = pd.read_csv('MULTPL-SP500_PE_RATIO_MONTH.csv')
index_indiv = pd.read_csv('YALE-US_CONF_INDEX_1YR_INDIV.csv')
index_inst = pd.read_csv('YALE-US_CONF_INDEX_1YR_INST.csv')
dip_buying_indiv = pd.read_csv('YALE-US_CONF_INDEX_BUY_INDIV.csv')
dip_buying_inst = pd.read_csv('YALE-US_CONF_INDEX_BUY_INST.csv')
crash_indiv = pd.read_csv('YALE-US_CONF_INDEX_CRASH_INDIV.csv')
crash_inst = pd.read_csv('YALE-US_CONF_INDEX_CRASH_INST.csv')
valuation_indiv = pd.read_csv('YALE-US_CONF_INDEX_VAL_INDIV.csv')
valuation_inst = pd.read_csv('YALE-US_CONF_INDEX_VAL_INST.csv')
datasets = [spy, pe, index_indiv, index_inst, dip_buying_indiv, dip_buying_inst, crash_indiv, crash_inst, valuation_indiv, valuation_inst]


spy['Date'] = pd.to_datetime(spy['Date'], errors='coerce')
spy = spy.dropna()

for df in datasets[1:]:
    df['Date_unused'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna()
for df in datasets[1:]:
    index_indiv['Date_unused'] = index_indiv['Date_unused'].map(lambda x: pd.datetime(int(x.year), int(x.month) - 1, int(1)) if int(x.month) != 1 else pd.datetime(int(x.year), 12, 1))


res = pd.merge(spy.assign(grouper=spy['Date'].dt.to_period('M')),
               pe.assign(grouper=pe['Date_unused'].dt.to_period('M')),
               how='left', on='grouper')


for df in datasets[2:]:
    res = pd.merge(res.assign(grouper=spy['Date'].dt.to_period('M')),
               df.assign(grouper=pe['Date_unused'].dt.to_period('M')),
               how='left', on='grouper')

useful_columns = ['Date_x', 'Open', 'High', 'Low', 'Close', 'Volume', 'PE', 'index_conf_indiv',
        'index_inst', 'index_dip_indiv', 'index_dip_inst', 'index_indiv_crash',
       'index_inst_crash', 'index_indiv_val', 'index_inst_val']

res = res[useful_columns]
res = res.loc[:, ~res.columns.duplicated()]

res['next_open'] = res['Open'].shift(-1)
res['next_close'] = res['Close'].shift(-1)
res['percent_change_predictor'] = (res['next_close'] - res['next_open'])/res['next_open']
res.loc[res['percent_change_predictor'] <= 0, 'direction_next_day'] = 0
res.loc[res['percent_change_predictor'] > 0, 'direction_next_day'] = 1
res['today_change'] = 100 * (res['Close'] - res['Open'])/res['Open']
res['max_change'] = 100 * (res['High'] - res['Low'])/res['High']

useful_columns_after_math = ['direction_next_day','today_change', 'max_change', 'Volume', 'PE', 'index_conf_indiv',
        'index_inst', 'index_dip_indiv', 'index_dip_inst', 'index_indiv_crash',
       'index_inst_crash', 'index_indiv_val', 'index_inst_val']

    
res = res[useful_columns_after_math]
#thesis is that people will either buy the dip or will sell the next day to cool the market if stocks climb


final_df = res.loc[abs(res['today_change']) > 0.9]
final_df=(final_df-final_df.min())/(final_df.max()-final_df.min())

#correlation matrix to help weed out some of the variables

print(final_df.corr())
sns.heatmap(final_df.corr(), cmap="YlGnBu", annot=True)
plt.show()
final_df = final_df.dropna()
from statsmodels.stats.outliers_influence import variance_inflation_factor
X_variables = final_df[useful_columns_after_math[1:]]
vif_data = pd.DataFrame()
vif_data["feature"] = X_variables.columns
vif_data["VIF"] = [variance_inflation_factor(X_variables.values, i) for i in range(len(X_variables.columns))]

useful_columns_after_vif = ['direction_next_day','today_change', 'max_change', 'Volume', 'PE',
        'index_inst', 'index_dip_indiv', 'index_dip_inst', 'index_indiv_crash',
       'index_inst_crash', 'index_inst_val']

res = res[useful_columns_after_vif]
#thesis is that people will either buy the dip or will sell the next day to cool the market if stocks climb


final_df = res.loc[abs(res['today_change']) > 0.9]
final_df=(final_df-final_df.min())/(final_df.max()-final_df.min())

#correlation matrix to help weed out some of the variables

# print(final_df.corr())
sns.heatmap(final_df.corr(), cmap="YlGnBu", annot=True)
plt.show()
final_df = final_df.dropna()
from statsmodels.stats.outliers_influence import variance_inflation_factor
X_variables = final_df[useful_columns_after_vif[1:]]
vif_data = pd.DataFrame()
vif_data["feature"] = X_variables.columns
vif_data["VIF"] = [variance_inflation_factor(X_variables.values, i) for i in range(len(X_variables.columns))]
# print(vif_data)



useful_columns_after_vif_round_2 = ['direction_next_day','today_change', 'max_change', 'Volume', 'PE', 'index_dip_indiv', 'index_dip_inst',
       'index_inst_crash', 'index_inst_val']

X_variables = final_df[useful_columns_after_vif_round_2[1:]]
vif_data = pd.DataFrame()
vif_data["feature"] = X_variables.columns
vif_data["VIF"] = [variance_inflation_factor(X_variables.values, i) for i in range(len(X_variables.columns))]
print(vif_data)

res = res[useful_columns_after_vif_round_2]
#thesis is that people will either buy the dip or will sell the next day to cool the market if stocks climb


final_df = res.loc[abs(res['today_change']) > 0.9]
final_df=(final_df-final_df.min())/(final_df.max()-final_df.min())

#correlation matrix to help weed out some of the variables

sns.heatmap(final_df.corr(), cmap="YlGnBu", annot=True)
plt.show()
final_df = final_df.dropna()
from statsmodels.stats.outliers_influence import variance_inflation_factor

X_variables = final_df[useful_columns_after_vif_round_2[1:]]
vif_data = pd.DataFrame()
vif_data["feature"] = X_variables.columns
vif_data["VIF"] = [variance_inflation_factor(X_variables.values, i) for i in range(len(X_variables.columns))]
# print(vif_data)

y = final_df[useful_columns_after_vif_round_2[0]]   
x = X_variables

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)



zero = 0
one = 1
for val in y_test:
    if val == 0:
        zero += 1
    else:
        one += 1

print(zero, one)

LogReg = LogisticRegression()
LogReg.fit(x_train, y_train)
LogReg_pred = LogReg.predict(x_test)
print("logistic regression accuracy score is " + str(accuracy_score(y_test, LogReg_pred)))
print(f1_score(y_test, LogReg_pred, average='macro'))
print(matthews_corrcoef(y_test, LogReg_pred))

SVC_model = svm.SVC()
SVC_model.fit(x_train, y_train)
SVC_prediction = SVC_model.predict(x_test)
print("svm accuracy score is " + str(accuracy_score(SVC_prediction, y_test)))
print(f1_score(y_test, SVC_prediction, average='macro'))
print(matthews_corrcoef(y_test, SVC_prediction))

KNN_model = KNeighborsClassifier(n_neighbors=5)
KNN_model.fit(x_train, y_train)
KNN_prediction = KNN_model.predict(x_test)
print("knn accuracy score is " + str(accuracy_score(KNN_prediction, y_test)))
print(f1_score(y_test, KNN_prediction, average='macro'))
print(matthews_corrcoef(y_test, KNN_prediction))

naive_bayes = GaussianNB()
naive_bayes.fit(x_train, y_train)
naive_bayes_prediction = naive_bayes.predict(x_test)
print("Naive Bayes accuracy is " + str(accuracy_score(naive_bayes_prediction, y_test)))
print(f1_score(y_test, naive_bayes_prediction, average='macro'))
print(matthews_corrcoef(y_test, naive_bayes_prediction))

minDepth = 100
minRMSE = 100000
for depth in range(2,20):
  tree_reg = DecisionTreeRegressor(max_depth=depth)
  tree_reg.fit(x_train, y_train)
  y_pred = tree_reg.predict(x_test)
  mse = mean_squared_error(y_test, y_pred)
  rmse = np.sqrt(mse)
  
  if rmse < minRMSE:
    minRMSE = rmse
    minDepth = depth

tree_model = tree.DecisionTreeClassifier(max_depth=minDepth)
# Fit a decision tree
tree_model = tree_model.fit(x_train, y_train)
# Training accuracy
tree_prediction = tree_model.predict(x_test)
print("Decision Tree accuracy is " + str(accuracy_score(tree_prediction, y_test)))
print(f1_score(y_test, tree_prediction, average='macro'))
print(matthews_corrcoef(y_test, tree_prediction))


random_forest=RandomForestClassifier(n_estimators=100)
random_forest.fit(x_train,y_train)
random_forest_pred = random_forest.predict(x_test)
print("Random Forest accuracy is " + str(accuracy_score(random_forest_pred, y_test)))
print(f1_score(y_test, random_forest_pred, average='macro'))
print(matthews_corrcoef(y_test, random_forest_pred))

