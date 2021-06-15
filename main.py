# %% [markdown]
# # Project C116
# %% [markdown]
# ## Getting Data

# %%
import pandas

data_frame = pandas.read_csv("https://raw.githubusercontent.com/whitehatjr/datasets/master/pro-c116/Admission_Predict.csv")

toefl_score = data_frame["TOEFL Score"].to_list()
gre_score = data_frame["GRE Score"].to_list()
coa = data_frame["Chance of admit"].to_list()

# %% [markdown]
# ## Showing Data

# %%
import plotly.express as px

figure = px.scatter(x=toefl_score, y=gre_score, color=coa, labels=dict(x="TOEFL Score", y="GRE Score", color="Chance of Admitting"), title="TOEFL Score vs GRE Score")

figure.update_traces(marker=dict(line=dict(color='DarkSlateGrey')))

figure.show()

# %% [markdown]
# ## Train Test Split

# %%
from sklearn.model_selection import train_test_split

scores = data_frame[["TOEFL Score", "GRE Score"]]

scores_train, scores_test, coa_train, coa_test = train_test_split(scores, coa, test_size=0.25, random_state=0)

# %% [markdown]
# ## Logistic Regression

# %%
from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(random_state=0)
log_reg.fit(scores_train, coa_train)

# %% [markdown]
# ## Prediction Accuracy

# %%
from sklearn.metrics import accuracy_score

prediction = log_reg.predict(scores_test)
accuracy = accuracy_score(coa_test, prediction)

print(f"Accuracy of prediction: {accuracy}")


