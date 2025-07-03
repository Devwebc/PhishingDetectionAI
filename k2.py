
import pandas as pd
import warnings
from IPython.display import display
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import plotly.express as px
import plotly.express as px
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings('ignore')

#cleanse the data

df = pd.read_csv('Data\spam.csv',encoding='latin-1')
df.head()
print(" Total number of rows in the dataset are", len(df))
#drop the non-relevant unnamed columns 
df=df.drop(['Unnamed: 2', 'Unnamed: 3','Unnamed: 4'],axis=1)
# Renaming v1 & v2 as Category & Text
df=df.rename(columns={"v1":"Category","v2":"Text"})
#create a column to check of each text & plot a histogram to check the distirbution
df['Length']=df['Text'].apply(len)
display(df.head())
#distribution of the data
fig = px.histogram(df, x='Length', marginal='rug',
                   title='Histogram of Text Length')
fig.update_layout(
    xaxis_title='Length',
    yaxis_title='Frequency',
    showlegend=True)


fig = px.histogram(df, x='Length', color='Category', marginal='rug',
                   title='Histogram of Text Length by Category')
fig.update_layout(
    xaxis_title='Length',
    yaxis_title='Frequency',
    showlegend=True)
fig.show()
#Let's Label the data as 0 & 1 i.e. Spam as 1 & Ham as 0
df.loc[:,'Category']=df.Category.map({'ham':0, 'spam':1})
df['Category'] = df['Category'].astype(int)
df.head()

#Train
count = CountVectorizer()
text = count.fit_transform(df['Text'])
#Train & test split
x_train, x_test, y_train, y_test = train_test_split(text, df['Category'], test_size=0.30, random_state=100)
text
#Let's print the dimentions of the train & test dataset
display('X-Train :', x_train.shape)
display('X-Test :',x_test.shape)
display('Y-Train :',y_train.shape)
display('X-Test :',y_test.shape)
mlp_classifier_model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000) 
mlp_classifier_model.fit(x_train, y_train)  
prediction = mlp_classifier_model.predict(x_test)
# Calculate and print classification metrics
print("MLP Classifier")
print("Accuracy score: {:.2f}".format(accuracy_score(y_test, prediction)))
print("Precision score: {:.2f}".format(precision_score(y_test, prediction)))
print("Recall score: {:.2f}".format(recall_score(y_test, prediction)))
print("F1 score: {:.2f}".format(f1_score(y_test, prediction)))

# Save fitted vectorizer and trained model
#joblib.dump(count, "vectorizer.pkl")
#joblib.dump(mlp_classifier_model, "mlp_model.pkl")

#Test
#new_message = [" “Congratulations - you're a winner! Go to bit.ly/eFgHiJK to claim your $500 Walmart gift card.”"]
# Transform the new message using the same CountVectorizer
#new_message_transformed = count.transform(new_message)
# Predict
#new_prediction = mlp_classifier_model.predict(new_message_transformed)
#print("Prediction:", new_prediction[0])  # 0 for ham, 1 for spam
