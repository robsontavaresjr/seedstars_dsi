import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt


dados  = pd.read_csv('Data Science Case Study - Data.csv', thousands='.', decimal=',')
dados = dados[['label', 'Education_Status', 'Gender', 'Age', 'No_Children', 'referrals', 'Amount', 'Reason']]

#Cleaning Amount
dados.loc[dados['Amount'] == '-', 'Amount'] = 0
dados['Amount'] = dados['Amount'].apply(lambda x: float(x))
dados.loc[dados['Amount'] < 10, 'Amount'] = dados.loc[dados['Amount'] < 10, 'Amount'].apply(lambda x: x*1000)
dados.loc[dados['Amount'] < 100, 'Amount'] = dados.loc[dados['Amount'] < 100, 'Amount'].apply(lambda x: x*100)
dados.loc[dados['Amount'] > 100, 'Amount'] = dados.loc[dados['Amount'] > 100, 'Amount'].apply(lambda x: x*10)
dados.loc[dados['Amount'] == 0, 'Amount'] = dados.loc[dados['Amount'] != 0, 'Amount'].mean()

#Cleaning Age data
dados.loc[dados['Age'] < 1, 'Age'] = dados.loc[dados['Age'] < 1, 'Age'].apply(lambda x: x*100)
dados.loc[dados['Age'] < 10, 'Age'] = dados.loc[dados['Age'] < 10, 'Age'].apply(lambda x: x*10)
dados['Age'].fillna(int(dados['Age'].mean()), inplace=True)
dados.loc['Age'] = dados.loc[dados['Age'] > 18, 'Age']

dados = dados.loc[dados.index != 'Age']

scaler = preprocessing.StandardScaler()
dados[['Age', 'Amount']] = scaler.fit_transform(dados[['Age', 'Amount']].values)

# Enconding categorical Variables
dadosLimpos = dados.copy()
dadosLimpos.Gender = dados.Gender.apply(lambda x: list(dados.Gender.drop_duplicates().values).index(x))
dadosLimpos.Reason = dados.Reason.apply(lambda x: list(dados.Reason.drop_duplicates().values).index(x))

#Splitting data
X = dadosLimpos[['Education_Status', 'Gender', 'Age', 'No_Children',
       'referrals', 'Amount', 'Reason']].values

y = dadosLimpos.label.values

X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.33, random_state=42)

#Predicting with Logistic Regression
logReg = linear_model.LogisticRegression()

model = logReg.fit(X_train, y_train)

#Results
y_pred = logReg.predict(X_test)

#Scoring the results to prove the model was a good fit

#Accuracy
print("Accuracy is: %.2f"  % logReg.score(X_test, y_test))

#Confusion Matrix
print("Confusion Matrix: \n")
print(confusion_matrix(y_test, y_pred))

#Precision
print("Classification Report: \n")
print(classification_report(y_test, y_pred))

#ROC
logit_roc_auc = roc_auc_score(y_test, logReg.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logReg.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()


