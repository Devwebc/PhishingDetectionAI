import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report


data = {
    'text': [
        "Congratulations! You've won a $1000 gift card. Claim now!",
        "Hi, I hope you're doing well. Just checking in.",
        "Urgent: Your account has been compromised. Click here to verify.",
        "Reminder: Your invoice is due. Please find attached.",
        "Important: Update your payment information to avoid service interruption.",
        "Hello! Your new package has been delivered.",
        "Great news! You've been selected for a special offer. Click here.",
        "Friendly reminder: Your subscription is expiring soon.",
        "Final notice: Pay your bill today to avoid late fees.",
        "Hi there! Do you need help with your project?",
        "Congratulations! You’ve won a free vacation package!",
        "Update your profile now to receive special offers.",
        "Important: Your action is required to complete your registration.",
        "Hello, I am reaching out to schedule our next meeting.",
        "Alert: Your password needs to be reset. Click here for more info.",
        "Congratulations! You've been pre-approved for a loan.",
        "Reminder: Your monthly payment is due. Don't forget to pay.",
        "Hi, just wanted to confirm your meeting for tomorrow.",
        "Last chance! Claim your prize now before it expires.",
        "You have a new message from the HR department."
    ],
    'label': [1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0]
}




df = pd.DataFrame(data)

print('Dataset: ')
print(df)

X = df['text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_vectorized,y_train)

y_pred = model.predict(X_test_vectorized)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"\n Accuracy: {accuracy: .2f}")
print("Classification Report:")
print(report)

