from flask import Flask, request
import pandas as pd
import numpy as np
import pickle
import flasgger
from flasgger import Swagger

app=Flask(__name__)
Swagger(app)


pickle_in=open('pipe.pkl','rb')
DecisionTreeClassifier=pickle.load(pickle_in)

@app.route('/')
def welcome():
    return "Welcome to the world of titanic"


@app.route('/predict',methods=["Get"])
def predict_survived():
    
    """Let's deep dive into the world of titanic 
    This is using docstrings for specifications.
    ---
    parameters:  
      - name: Pclass
        in: query
        type: number
        required: true
      - name: Sex
        in: query
        type: string
        required: true
      - name: Age
        in: query
        type: number
        required: true
      - name: SibSp
        in: query
        type: number
        required: true
      - name: Parch
        in: query
        type: number
        required: true  
      - name: Fare
        in: query
        type: number
        required: true
      - name: Embarked
        in: query
        type: string
        required: true
    
    responses:
        200:
            description: The output values
    """

    Pclass=request.args.get("Pclass")
    Sex=request.args.get("Sex")
    Age=request.args.get("Age")
    SibSp=request.args.get("SibSp")
    Parch=request.args.get("Parch")
    Fare=request.args.get("Fare")
    Embarked=request.args.get("Embarked")
    prediction=DecisionTreeClassifier.predict([[Pclass,Sex,Age,SibSp,Parch,Fare,Embarked]])
    print(prediction)
    return "Hello the answer is"+str(prediction)

if __name__=='__main__':
    app.run()
       
        
    