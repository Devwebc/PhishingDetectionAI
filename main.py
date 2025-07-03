
from fastapi import FastAPI,Response
from fastapi.responses import FileResponse, HTMLResponse
import joblib
from k3 import detect

app = FastAPI()

@app.get("/favicon.ico", include_in_schema=False)
def remove_favicon():
    return Response(status_code=204)  # No Content

@app.get("/")
def read_root():
    #return {"Hello": "World"}
     return FileResponse("insert.html")
 
 
count = joblib.load("vectorizer.pkl")
mlp_classifier_model = joblib.load("mlp_model.pkl")

    
@app.get("/detect")
def detect_email(email: str, sender: str):
    new_message_transformed = count.transform([email])  # ✅ Wrap in list
    new_prediction = mlp_classifier_model.predict(new_message_transformed)
    t=['chimene@company.com','ceo@company.com','example@company.com']
    server="company.com"

    if str(new_prediction[0]) == '1':
        new_pred = 'Spam'
    elif str(new_prediction[0]) == '0': 
        new_pred = 'Not Spam'
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Prediction Result</title>
    </head>
    <body>
        <div class="result-box">
            <h2>Prediction Result</h2>
            <div class="result-line"><span class="label">Email:</span> {email}</div>
            <div class="result-line"><span class="label">Sender:</span> {sender}</div>
            <div class="result-line"><span class="label">Email Content:</span> {str(new_pred)}</div>
            <div class="result-line"><span class="label">Sender:</span> {detect(sender,t,server)}</div>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)
