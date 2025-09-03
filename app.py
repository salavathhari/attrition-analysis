import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

st.set_page_config(page_title="Employee Attrition Dashboard", layout="wide")

st.title("📊 Employee Attrition Analysis – Green Destinations")

@st.cache_data
def load_data():
    return pd.read_csv("greendestination.csv")

df = load_data()
st.write("### Dataset Preview")
st.dataframe(df.head())

# Convert target
df['Attrition'] = df['Attrition'].map({'Yes':1,'No':0})

# Overall Attrition Rate
attrition_rate = df['Attrition'].mean()*100
st.metric("Overall Attrition Rate", f"{attrition_rate:.2f}%")

# Sidebar filters
st.sidebar.header("Filters")
age_min, age_max = int(df['Age'].min()), int(df['Age'].max())
age_range = st.sidebar.slider("Age Range", age_min, age_max, (age_min, age_max))
income_min, income_max = int(df['MonthlyIncome'].min()), int(df['MonthlyIncome'].max())
income_range = st.sidebar.slider("Income Range", income_min, income_max, (income_min, income_max))

filtered = df[(df['Age'].between(*age_range)) & (df['MonthlyIncome'].between(*income_range))]

st.write("### Filtered Data", filtered.shape[0], "employees selected")
st.dataframe(filtered.head())

# Attrition by OverTime
st.write("### Attrition by OverTime")
ct = pd.crosstab(df['OverTime'], df['Attrition'])
fig, ax = plt.subplots()
ct.plot(kind='bar', stacked=True, ax=ax)
plt.title("OverTime vs Attrition")
st.pyplot(fig)

# Logistic Regression Model
st.write("### Logistic Regression Model")
features = ['Age','YearsAtCompany','MonthlyIncome','JobSatisfaction','OverTime']
X = df[features].copy()
X = pd.get_dummies(X, columns=['OverTime'], drop_first=True)
y = df['Attrition']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
model=LogisticRegression(max_iter=2000)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
proba=model.predict_proba(X_test)[:,1]

st.text("Confusion Matrix")
st.write(confusion_matrix(y_test,y_pred))
st.text("Classification Report")
st.text(classification_report(y_test,y_pred))
st.metric("ROC-AUC", f"{roc_auc_score(y_test, proba):.2f}")

# Feature importance
coefs = pd.Series(model.coef_[0], index=X.columns).sort_values()
fig2, ax2 = plt.subplots()
coefs.plot(kind='barh', ax=ax2)
plt.title("Feature Importance (Logistic Regression Coefficients)")
st.pyplot(fig2)

st.success("Dashboard loaded successfully! Use sidebar filters to explore data.")
