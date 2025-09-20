import uvicorn, pickle
from fastapi import FastAPI
from pydantic import BaseModel
# advantage of using pydantic is that it provides data validation and parsing
# It ensures that the data received in the API requests adheres to the specified types and constraints

class BankNote(BaseModel):
    variance : float 
    skewness : float 
    curtosis : float 
    entropy  : float
# ----------------------------------------------------------------------------
# 1. Create the app object
app = FastAPI()

pickle_in  = open("../Output/model.pkl", "rb")
classifier = pickle.load(pickle_in)

# ----------------------------------------------------------------------------
@app.get('/')
def index():
    return {'message': 'Hello, World'}

@app.get('/{name}')
def get_name(name: str):
    return {'Welcome ': f'{name}'}

# ----------------------------------------------------------------------------
# 2. Expose the prediction functionality, make a prediction from the passed
#    JSON data and return the predicted Bank Note with the confidence
@app.post('/predict')
def predict_banknote(data:BankNote):
    data     = data.dict()
    variance = data['variance']
    skewness = data['skewness']
    curtosis = data['curtosis']
    entropy  = data['entropy']
    
    prediction = classifier.predict([[variance,skewness,curtosis,entropy]])
    if(prediction[0]>0.5):
        prediction="Fake note"
    else:
        prediction="Its a Bank note"
    return {
        'prediction': prediction
    }

# ----------------------------------------------------------------------------
# 3. Run the API with uvicorn
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)