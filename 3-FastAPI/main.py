import uvicorn              # ASGI server
from fastapi import FastAPI
# ----------------------------------------------------------------------------
# 1. Create the app object                                                    |
# ----------------------------------------------------------------------------
app = FastAPI()

# ----------------------------------------------------------------------------
# 2. Index route, opens automatically on http://127.0.0.1:8000                |
# ----------------------------------------------------------------------------
@app.get('/')               # decorator says: Anyone visits / (main) using GET → executes the index function.
def index():
    return {'message': 'Hello, World'}

# ----------------------------------------------------------------------------
# 3. Route with a single parameter, returns the parameter within a message    |
#    Located at: http://127.0.0.1:8000/AnyNameHere                            |
# ----------------------------------------------------------------------------
@app.get('/Welcome')        # http://127.0.0.1:8000/Welcome?name=Abdelrhman    → {"Welcome": "Abdelrhman"}
def get_name(name: str):
    return {'Welcome': f'{name}'}

# ----------------------------------------------------------------------------
# 4. Run the API with uvicorn                                                 |
#    Will run on http://127.0.0.1:8000                                        |
# ----------------------------------------------------------------------------
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)           # python main.py   → runs the app

# ----------------------------------------------------------------------------
# Recommended To run the app, use the command → uvicorn main:app --reload     |
# ------------                                                                |
# main      : the file name "main.py" without the .py extension.              |
# app       : the object created in the file main.py.                         |
# --reload  : make the server restart after code changes.                     |
# ----------------------------------------------------------------------------