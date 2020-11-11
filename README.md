# Credit Default Detector

This project shows multiple Jupyter Notebooks which demonstrate a variety of Machine Learning Classification models built and tuned to detect users that are likely to default credit loan for the [UCI Credit Card Clients](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients) dataset. 

The repository is also served as a hub of different deployment techniques using FastAPI and Unicorn: (1) manual input and (2) upload file

* Jupyter Notebooks: (1) traditional-ml model utilized the traditional algorithms (logistis regression, kNN, SVC, decision tree, random forest) while (2) mlp and nn models leveraged the applicability of neural networks, which produced the highest accuracy among all (85%).
* main.py (+ dependency files/folder): it is served as a deployment document using FastAPI that enables us to upload data file into the server and make prediction at hands. 
* manual input: functions similar as above, the deployment package, that,  however, provides the option of manually inputting the data on the server and make prediction (single output only).

## Requirements

* pandas
* numpy
* matplotlib
* seaborn
* sklearn
* tensorflow
* fastapi
* uvicorn
