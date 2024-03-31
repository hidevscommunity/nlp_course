**Chapter 8 -** **Model Deployment**

Once the expectations from machine learning are achieved, the next big step is putting the built model in production i.e. making it available to the end-users/clients. The model deployment also completes the data science/machine learning project life cycle.

**Why do we need to deploy an ML model?**

Most of the machine learning projects are a value addition or augmentation to an existing service/solution or product. So a built ML model adds value to the organization only if its insights/predictions are made available to the end-users/systems. Therefore, it becomes necessary to deploy ML models.

_"No machine learning model is valuable unless it’s deployed to production." – Luigi Patruno_ 

Generally, models in python are deployed using -

- **Flask -** Flask is a lightweight web application framework written in Python. Using Flask, we can easily integrate the ML models into an existing(or new) environment where it takes the user inputs, passes them to the ML pipeline and displays(or utilizes further) the predictions to the end-users/clients.

- **Streamlit** - Streamlit allows to quickly create highly interactive data apps/web applications without having any prior (or minimal) knowledge about web designing and web services framework. With few lines of code, models can be deployed using streamlit.

**Model deployment using Flask**

First we need to install the _Flask_ python library using the _pip install flask_ command. The complete deployment archive/folder can be downloaded from [here](https://drive.google.com/file/d/1WTz5aWzn1h3Pr7CJ6YiUA7k1XSQ6PVIZ/view?usp=share_link).

The above folder’s structure looks like this.

![](https://lh7-us.googleusercontent.com/WWs7JzQjk5-kGq788-tq6vhyZbELuAY2B0xIPi7jnsAzG9R5H5N6TLc0_l_04oxZq8UXVk7ESAgpKVR4sEmDlG6tYHT5H-o-I47DwgvWuV2M7SSgcHMGSRtBAuS9mafz09DmhFul0yL9lPRDUuWr79A)

Here - 

- **app.py** - This is the core script of the deployment. It receives the user’s input, processes it, passes it to the model and finally returns the classified news category.

- **models**  - A folder having pre-trained linear SVM and Word2Vec models.

- **news\_classification.py** - Since the input text needs to be preprocessed and featurized before passing to the model. So this file contains the same **preprocess\_text()**, **decontract()** **featurization()** and **classify()** functions that we have already seen earlier.

- **templates** - This folder contains the HTML templates to display the web pages as we interact with the application.  

- **requirements.txt** - it contents all necessary packages which used to run the application

**Code Explanation - app.py**

Different URLs can be routed to different functions in the flask. Which URL to be rendered on the webpage is specified using the _route_ function.  

Here, we have two _routes_ in our application.

1. A **home** page _route_ allows the users to input the article text.

2. A **predict** _route_ to perform news classification using the saved model.

Once the main URL ( <http://localhost:5002/> ) is loaded locally, the flask acknowledges the GET request and renders the _index.html_ template.

The _index.html_ template displays a heading titled “News Category Classification” and allows the user to type/insert the article text. Using _request. form_ functionality, flask extracts the user's input from the form.

Once the user types the article text and clicks on the _Submit_ button, the flask acknowledges the POST request, extracts the user’s input, passes it to the same model and displays the results by rendering the _results.html_ template.

**Running the app**

This application is hosted in local mode. To run the application, type the below command in the terminal from the _flask\_deployment_ folder.

python app.py __

The output looks like this as shown below.

![](https://lh7-us.googleusercontent.com/Ff34boNU3n4aZjEva3ocVoQiYlL1jy2hDqV5Ee_l0_b3DL6S8Y8eMYofL0ppLKYnVt0CngAgd_St4QZqyp42ATtfWD2Eje07KGMa7eMvhy_ZYTZiTdPwg0PjOAySacSao5O67rc4MFS8z9iRaZsr-aU)

Now access the base URL ( <http://localhost:5002/> ) in the browser, the home page looks like this 

![](https://lh7-us.googleusercontent.com/3qVAKQuGJOw0TCo5RnD2shyxRhisQWkve1hfEw_sHDFDRP1JpxtQ2Jgp-mbBhBTts9RrJeyXlEM2ymL6YAdw6YKk7fZHr1tiRLvwurh5F_QZrcUFBI6rjUdL908wOPMjdK4MzPNenVD07AM40hcnH_4)

****

The finally deployed application demo with some unseen input text articles is as shown below.

![](https://lh7-us.googleusercontent.com/WFPNfpqyhUjyzc96yZETkfOmnzriTJMKbeRr9ob7aQacKNE4fLivZ5Ef9qS4nSJMPaY59zZ2qpKo8PxnT2r3XEWha7IWkpaTKJjS4xp4BsT-hlvfoGXmjjw14EvrkEujVl9VP1vaPvOl0j6JVPJrRhk)
