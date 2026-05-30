import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Import Matplotlib an Seaborn for visualization
import matplotlib.pyplot as plt
import seaborn as sns


# Import dataset

train_df = pd.read_csv('./train.csv')
test_df = pd.read_csv('./test.csv')

print('Data successfully load')

# Print out of the shape of each dataset
print(f'Train dataset rows: {train_df.shape[0]}')
print(f'Test dataset rows: {test_df.shape[0]}')


df = train_df.drop(columns=['PassengerId', 'Name', 'Ticket'])
#df_encoded = pd.get_dummies(df, columns=['Sex', 'Embarked'])
df_clean = df
df_clean.dropna(subset=['Age'], inplace=True)


# Survival rates
print('Cabin_nan: ', df_clean['Cabin'].isna().sum())
df_clean['Cabin'] = df_clean['Cabin'].fillna('X')
df_clean['Cabin'] = df_clean['Cabin'].str[0]

surv0 = df_clean['Survived'] == 0
surv1 = df_clean['Survived'] == 1

plt.figure()
plt.subplot(3, 3, 1)
df_clean[surv0]['Pclass'].hist(alpha=0.5, label='Dead')
df_clean[surv1]['Pclass'].hist(alpha=0.5, label='Surv')
plt.legend()
plt.xlabel('Pclass')
plt.ylabel('#')

plt.subplot(3, 3, 2)
df_clean[surv0]['Sex'].hist(alpha=0.5, label='Dead')
df_clean[surv1]['Sex'].hist(alpha=0.5, label='Surv')
plt.legend()
plt.xlabel('Sex')
plt.ylabel('#')

plt.subplot(3, 3, 3)
min_ = df_clean['Age'].min()
max_ = df_clean['Age'].max()
bins = np.linspace(min_, max_, 10)
df_clean[surv0]['Age'].hist(alpha=0.5, bins=bins, label='Dead')
df_clean[surv1]['Age'].hist(alpha=0.5, bins=bins, label='Surv')
plt.legend()
plt.xlabel('Age')
plt.ylabel('#')

plt.subplot(3, 3, 4)
min_ = df_clean['SibSp'].min()
max_ = df_clean['SibSp'].max()
bins = np.linspace(min_, max_, 10)
df_clean[surv0]['SibSp'].hist(alpha=0.5, bins=bins, label='Dead')
df_clean[surv1]['SibSp'].hist(alpha=0.5, bins=bins, label='Surv')
plt.legend()
plt.xlabel('Sibsp')
plt.ylabel('#')

plt.subplot(3, 3, 5)
min_ = df_clean['Parch'].min()
max_ = df_clean['Parch'].max()
bins = np.linspace(min_, max_, 10)
df_clean[surv0]['Parch'].hist(alpha=0.5, bins=bins, label='Dead')
df_clean[surv1]['Parch'].hist(alpha=0.5, bins=bins, label='Surv')
plt.legend()
plt.xlabel('ParCh')
plt.ylabel('#')

plt.subplot(3, 3, 6)
min_ = df_clean['Fare'].min()
max_ = df_clean['Fare'].max()
bins = np.linspace(min_, max_, 10)
df_clean[surv0]['Fare'].hist(alpha=0.5, bins=bins, label='Dead')
df_clean[surv1]['Fare'].hist(alpha=0.5, bins=bins, label='Surv')
plt.legend()
plt.xlabel('Fare')
plt.ylabel('#')

plt.subplot(3, 3, 7)
df_clean[surv0]['Cabin'].hist(alpha=0.5, label='Dead')
df_clean[surv1]['Cabin'].hist(alpha=0.5, label='Surv')
plt.legend()
plt.xlabel('Cabin')
plt.ylabel('#')

plt.subplot(3, 3, 8)
df_clean[surv0]['Embarked'].hist(alpha=0.5, label='Dead')
df_clean[surv1]['Embarked'].hist(alpha=0.5, label='Surv')
plt.legend()
plt.xlabel('Embarked')
plt.ylabel('#')

plt.show(block=False)

df_clean = pd.concat([df_clean, pd.get_dummies(df_clean['Cabin'], prefix='Cabin', dtype=int)], axis=1)
df_clean = pd.concat([df_clean, pd.get_dummies(df_clean['Embarked'], prefix='Embarked', dtype=int)], axis=1)
df_clean = pd.concat([df_clean, pd.get_dummies(df_clean['Sex'], prefix='Sex', dtype=int)], axis=1)

# Correlation matrix
plt.figure()
df_clean_corr = df_clean[[
	'Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_male', 'Sex_female',
	'Cabin_T', 'Cabin_X', 'Embarked_C', 'Embarked_Q', 'Embarked_S']]
corr_mtx = df_clean_corr.corr()
sns.heatmap(corr_mtx, cmap='coolwarm')
plt.show(block=False)
print(df_clean.head(20))
print(df_clean.shape)

# Random forest classifier
df_clean = df_clean.drop(columns=['Sex', 'Embarked', 'Cabin'])
X = df_clean.drop(columns='Survived')
y = df_clean['Survived']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}\n")
print("Classification Report:")
print(classification_report(y_test, y_pred))





