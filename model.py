import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_confusion_matrix, classification_report, roc_curve, roc_auc_score

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 100)
plt.style.use('fivethirtyeight')

#load data
data = pd.read_csv('telco.csv')

#cleansing
kolom_inkonsisten = ['MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                    'StreamingTV', 'StreamingMovies']

nama_kolom = list(data.columns)
# print(nama_kolom)

for kolom in nama_kolom:
    data[kolom] = data[kolom].replace(' ', np.nan)

data['TotalCharges'] = data['TotalCharges'].astype('float64')
data['TotalCharges'] = data['TotalCharges'].fillna(data['TotalCharges'].median())

kolom_inkonsisten.pop(0)
data['MultipleLines'] = data['MultipleLines'].replace({'No phone service': 'No'})
for kolom in kolom_inkonsisten:
    data[kolom] = data[kolom].replace({'No internet service': 'No'})

#Data Preparation

kolom_object = list(data.select_dtypes('object').columns)
biner = []
non_biner = []

for kolom in kolom_object:
    if data[kolom].nunique()>2:
        non_biner.append(kolom)
    else:
        biner.append(kolom)
biner.pop()
le = LabelEncoder()

for kolom in biner:
    data[kolom] = le.fit_transform(data[kolom])

ohe = ['InternetService', 'PaymentMethod']

ordinal = {
    'Contract': {'Month-to-month': 0, 'One year': 1, 'Two year': 2},
    'Churn' : {'Yes': 1, 'No': 0}
}
data = data.replace(ordinal)

#one hot encoding
data = pd.get_dummies(data, columns=ohe)

data.drop('customerID', axis=1, inplace=True)

X = data.drop('Churn', axis=1)
y = data[['Churn']]
oversample = SMOTE()
X_smote, y_smote = oversample.fit_resample(X, y)

#modeling
model = RandomForestClassifier(criterion='entropy', n_estimators=200, n_jobs=-1)
model.fit(X_smote, y_smote)

# Saving model to disk
pickle.dump(model, open('model.pkl','wb'))
