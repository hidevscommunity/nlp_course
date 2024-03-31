**Chapter 7 - Text Classification – Classifying the news category**

Having studied the core NLP fundamentals and text featurization, let’s talk about text classification using machine learning and deep learning algorithms. 

Classification is a supervised machine learning approach where the target/output variable(Y) is already labeled with two or more classes. The ML algorithm learns from the data and classifies the class of new data points.

Based upon the number of classes in the target variable, classification tasks can be broadly classified into two categories.

1. **Binary Classification** - Having two classes. Examples - 

- Email spam detection(whether an email is a spam or ham)

- Fraud detection(whether a transaction is fraudulent or not)

- Rain prediction(whether it will rain today or not)

2. **Multi-Class classification** - Having more than two classes. Examples - 

- Sentiment Analysis(whether is sentiment is positive, negative or neutral)

- Human activity recognition(whether a person is sitting, standing, walking or lying down)

- Vehicle type detection(whether it’s a car, truck, bus, bike, train or aeroplane)

 Based on the above discussion we can quickly assess that our news category classification task is a multi-class(ternary specifically) classification task. 

There are multiple machine learning algorithms to solve a classification task, the most commonly used are - 

- Logistic Regression

- K-nearest Neighbors(KNN)

- Support Vector Machine(SVM)

- Decision Trees

- Random Forest                         

In this course, we are only going to discuss three major algorithms(Decision Tree, Logistic Regression and SVM). Before getting into them, let’s talk about training and test data.

****

**7.1 Train-test split**

Generally, the dataset is split into two parts before fitting an ML model.

- **Training set** - the subset of data to train an ML model. The model explores and learns from this data and makes predictions.

- **Test data** - the subset of data(apart from the training set) to evaluate the model’s performance. Generally, a model is tested against the test(new/unseen) data.  

The split ratio (like 80-20, 70-30 or something similar) depends on various factors like the type of task to be done, dataset available, domain knowledge etc. Here, we are going to split the featurized news articles data as per an 80-20 split ratio.

**train\_test\_split()** function from **sklearn.model\_selection** module is used to split the data into train-test sets. Since the ML model maps the input variables(predictors) to the target variable and the algorithm needs them to be specified in different parameters. Therefore, let’s keep the _category_ column in variable _y_.

|                                                                                                                                                                                                                                                                                            |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| from sklearn.model\_selection import train\_test\_split y = news\_df\["Category"] X\_train, X\_test, y\_train, y\_test = train\_test\_split(w2v\_summary, y, test\_size=0.2,random\_state = 42) print("Training set size :",X\_train.shape\[0]) print("Test set size :",X\_test.shape\[0]) |

Output - 

![](https://lh7-us.googleusercontent.com/UyPeeK9k8E7MMaWGWeke1uMAdgqMqdxG1mlsLw0kMsfoSMDRERWDGYonZQ5mhCV4NpZoUlhnmGpUOZYe8VyqNrm16JYj7JAeoh43J7vGNDEhfz_fJIIjSNnY53EmSJOYNwpsJjZ80oKjkS2aQNNNhEQ)

**7.2 Building a Machine Learning model**

Let’s start the ML model building process by discussing Decision tree first.

**Decision Tree**

As the name suggests, a decision tree(DT) is a tree-like structure having decision nodes and leaf nodes. Decision nodes are helpful in making decisions based on certain conditions whereas the leaf nodes contain the outcome(class labels). 

In order to predict an outcome, the tree is traversed from the root to the leaf node by following the corresponding branches as per the conditions.

Consider a case where the weather forecasting department provides weather updates along with chances of rainfall. The weather reports for 12 different cities are shown below.

****

|                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| City1 --> "The temperature is Mild, the humidity is High and the wind is Weak. High chances of rainfall." City2 --> "The temperature is Mild, the humidity is High and the wind is Strong. High chances of rainfall." City3 --> "The temperature is Hot, the humidity is High and the wind is Weak. High chances of rainfall." City4 --> "The temperature is Cold, the humidity is High and the wind is Weak. Less chances of rainfall." City5 --> "The temperature is Cold, the humidity is Normal and the wind is Strong. High chances of rainfall." City6 --> "The temperature is Hot, the humidity is Normal and the wind is Strong. High chances of rainfall." City7 --> "The temperature is Mild, the humidity is Normal and the wind is Weak. Less chances of rainfall." City8 --> "The temperature is Cold, the humidity is Normal and the wind is Weak. Less chances of rainfall." City9 --> "The temperature is Mild, the humidity is Normal and the wind is Strong. Less chances of rainfall." City10 --> "The temperature is Hot, the humidity is High and the wind is Strong. High chances of rainfall." City11 --> "The temperature is Hot, the humidity is Normal and the wind is Weak. High chances of rainfall." City12 --> "The temperature is Cold, the humidity is High and the wind is Strong. High chances of rainfall." |

Now if we tabulate these weather reports by considering the driver features(Temperature, Humidity, Wind and Rainfall), then the table looks like as shown below.

![](https://lh7-us.googleusercontent.com/7_7XraYucaWGCyr9agP2iI-i00vMgmXKffKcnGihrTFPXZrZr2VNIL9WLobX2OFwFwQ6uQI9p1qnEJVSbEu6BqZjHXF88aDpDcmxTqqwQBedhjtvjzoQ_oh-c69BWMiimtYzwyqN1TqJT1jnQOWbujk)

 If we try to predict the target variable (rainfall) using a decision tree by taking the features Temperature, Humidity, and Wind then the decision tree for this may look like as shown below.

![](https://lh7-us.googleusercontent.com/x2dz4cN1dZXEa6v0POI5NIdvGIPLPeC1IzONtNO2nx-Rm1s5O8rPrSqsqXDqRpWZM8DhmUXIfNvq_d1CDtDIFEQEJWqEwdXnm3-CPIDO0vdf6J5QLpJlO6yPmLABpo_yCQ6J_zXSUl_dA0b0gE6A5-E)

This above tree can be further simplified as -

****

|                                                                                                                                                                                                                                                                                                                          |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| if(Temperature == "Hot"):     rainfall = "Yes" elif(Temperature == "Mild"):     if(Humidity == "High")         rainfall = "Yes"     elif(Humidity == "Normal"):         rainfall = "No" elif(Temperature == "Cold"):     if(Wind == "Strong")         rainfall = "Yes"     elif(Wind == "Weak"):         rainfall = "No" |

So in simple terms, the decision tree is a nested if-else classifier.

While building a decision tree for a dataset with **_d_** features, the major challenge is which feature to place at the master node and at different other nodes. To solve this attribute selection problem, two statistical criteria are used.

- Information Gain   

- Gini Index

**a. Information Gain**  

As the name suggests, it measures how much information about the classes(target variable) can be gained using a feature. The feature with the highest information gain is taken at the root node and so on. 

Information gain further depends on **_Entropy_**. Entropy measures randomness in the data. Higher the entropy, the more difficult to draw conclusions from the information. 

Entropy for a dataset(S) with C classes is computed as 

![](https://lh7-us.googleusercontent.com/GtymUFisv-GOTkYiWeHbycg_v5dpn5VlfjoZTP1Scrx7muolRbzdPjQRX1FWAYaAfp8Xo6Po-VTz-OyYuoebDxDafriisB3zvi7lz3b8LxoZxezUeOXLUtSvsUWG7XYD4pitt0VwO70bLZEzidMWtUU)

where piis the probability of class _i_.

Similarly, Information gain is computed as

![](https://lh7-us.googleusercontent.com/boX6yuzvK3iySSRY4oMK4FKOR3JUYIjsvn_dzmR2lQOEloEjG8wXxrKKt7KuLmHdyqIeH1JVoY7WAPOkxWD9gl9rq0DO7gXHWrWTKJkdPWTjDJ9cav6LCJOo3voc_IHU4f4A86BQI6aO_TkcsbhmQZ4)

**b. Gini Index**

It measures how often a randomly picked datapoint would be incorrectly classified. So basically it is a measure of impurity and an attribute with a low Gini index is preferred first. Its computation formula is as shown below. 

![](https://lh7-us.googleusercontent.com/Xn6QvfKuCq5k6WZ_5BLyo-5Q2DCF5RF11M6XP59DpfOC8qGAfD8-B0OrcNnUrraD8v4tnuNV8trBDi0rbfJHTiMCYS1OqNxcGqzM0M4kfAjsgE63YyFm-01ZC6-c9OSD8l1IvE81dooFMFS0X3Z_iM4)

**Implementing decision tree in Python**

Decision tree is implemented using the **DecisionTreeClassifier__** class ****from **sklearn.tree** module. Further, fit() method is used to fit a DT model. 

****

|                                                                                                                                                                                        |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| from sklearn.tree import DecisionTreeClassifier dt\_classifier = DecisionTreeClassifier(random\_state = 42) #fitting decision tree model print(dt\_classifier.fit(X\_train, y\_train)) |

Output - 

![](https://lh7-us.googleusercontent.com/9kWDQDWtMCO1ev1yUnxceNrwmLwb7hfu_5T7Y6GL6tAZuL-W3FwonzT3djuZMWGG7_CvO06iBxvonvnJkcBsS6ZpiGV0HRV8ECCoCmxsViheC1y9X0I4q98btxpDz-qdi-NZFYPrvDWctx009ximeYY)

**7.3 Testing the model**

Having built a DT model, now it’s time to test it against the test data in order to check its usefulness.

predict() method is used to make predictions from a fitted model. Let’s compare the predicted and actual values for the first 15 observations.

First 15 actual news categories are

|                             |
| --------------------------- |
| print(y\_test\[:15].values) |

![](https://lh7-us.googleusercontent.com/mHS_7hcpuMPjxBMbRiGAl9VlITKraq22MhtNr-YJEgkluv7ZyUCZ7HIdUllSLaX8rTjg0m4w3BcmoITnYiXMYwc79GDWT17HOFpouKg9uQMkY6nFlmAVp5TqeBgEughP9YcnucbX3HD-JGqeKP_KES0)

First 15 predicted news categories are 

|                                                                                                                           |
| ------------------------------------------------------------------------------------------------------------------------- |
| y\_pred = dt\_classifier.predict(X\_test) #predictions on the first 15 observation from the test set print(y\_pred\[:15]) |

![](https://lh7-us.googleusercontent.com/oSUls-50Cjf0nVC1H1roLL0xyu4iExHwAy7bd5rsXTFlx7seHbiZZKPyfIijfRZ2MJnV7Kv0I-OLegc8hy3umotw4hHWvDt5EMVkxWrpW2eIxx0yJ9EsiFqPDcDQ4SoGdSmzl-_y8mvZbeYPoJ3FwME)

By comparing these, we can observe that only 12 categories are correctly classified and 3 are misclassified. For e.g. the first observation actually belongs to the “Business News” category but the DT model classifies it as the “World News” category.

In the next section, we will see how to evaluate a model using complete test data.

**7.4 Evaluating the model’s performance**

The most interesting part after a model building is the performance evaluation to assess its goodness and efficiency. Evaluation metrics(performance measures) are used for this purpose, basically, they quantify the performance of a model. 

**Why do we need to evaluate the models?**

In machine learning (or deep learning), there are multiple algorithms/approaches to solving a given problem. And we may end up in multiple models either from these algorithms or from the same algorithm (with different hyperparameter combinations). The biggest challenge over here is how to select the best model as per the business constraints and domain knowledge. 

So using some **evaluation metrics** we can compare these models and assess their efficiency by quantifying their performance. 

 There are various types of evaluation metrics for classification tasks. Most commonly used are 

1. Accuracy

2. Precision and Recall

3. F1-score

**1.  Accuracy**

Accuracy is the ratio of correctly classified observations to the total observations(total predictions made), i.e.

![](https://lh7-us.googleusercontent.com/UmuQK8J21l692ec2-w4y07tUHQYI1Cg7WaDmRq-x_-W5SlXZqm7pRAwORh3lhtZIQN1RKDAhlZTbY4QWelWXuPQ5xcniFqFURT9c-cFiCgfUcQ_abAbUE4dXXIi_0uEOnPTdhFkLqXAm7ss53EDsrOg)

**accuracy\_score()** from **sklearn.metrics** module is used to compute model accuracy in python.

|                                                                                                                                                                      |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| from sklearn.metrics import accuracy\_score dt\_accuracy = accuracy\_score(y\_true=y\_test, y\_pred=y\_pred) print("Accuracy using Decision Trees: ", dt\_accuracy)  |

Output - 

![](https://lh7-us.googleusercontent.com/mmQHPW14MHnNFhWDOz-Q_fxxWjy_SCO0I0TZdOsCNDbQhNraI8SVHFGEPpuALrs8JUYG7fv-FOExp8gPYPGNVCzQNcBd5u4Gw_LWWeiQxlbBSTeYYcbyhs_vPVPKpexw3xf6y0S4rx-uFWk3au55W6o)

So the built decision tree model on the news category dataset is 63.02% accurate.

**Caveats**

- Accuracy gives a false sense of evaluation while dealing with imbalanced data where most of the observation belongs to one class(majority class). 

Consider a case of cancer detection where out of 1000 people only 10 have cancer. If our model classifies every observation as Non-cancerous then the model is 99.90% accurate where 990 non-cancerous patients are correctly classified and 10 cancerous patients are classified as non-cancerous. 

Here the cost of misclassification of a minor class(cancerous) is very high, for e.g. classifying a cancer patient as non-cancerous is not acceptable as it's a matter of life and death.

Before discussing other evaluation measures, let’s first talk about the confusion matrix.

****

**Confusion Matrix**

Confusion matrix provides a more intuitive picture of the correct and incorrect classification counts for all the classes. Basically, it is an NxN matrix where N is the total number of classes. 

For a binary classification problem, the confusion matrix looks like

![](https://lh7-us.googleusercontent.com/qGOBrzkG2E1o4Q7lfiS5VzMg2uO2-pUiWq2xMmXB56mmgY_PrzJ1JLMyaIPUu4v8SsLbr2ybGmGrYg6UwMS5FixOx-HzqtPk_LzlaPPtIw243rxIQKzg2BEicJEXZtRRNmlb6nMp9tOMwyn0GhS7A0k) 

The terminologies used in the confusion matrix are

- **True Positive (TP)** - Actual class is positive and the model is also predicted as positive.

- **False Positive (FP) -** Actual class is negative but the model predicted as positive.

- **False Negative (FN) -** Actual class is positive but the model predicted as negative.

- **True Negative (TN)** - Actual class is negative and the model is also predicted as negative.

Accuracy using confusion matrix is defined as - 

![](https://lh7-us.googleusercontent.com/6mAs1PP48wpTfy6GTvWqT8o9LmyCbIdJZ634OpW5yUl61ibXskOg6FFMYmv1DpU7BrhHd38ObwSfUZ6p6YDm-PQZ8QPYlHC3BeESeq0nwvxWfbcjby9f9jcT7s4f7KD5wsEcVhWoUDMx2eDzFSwgFeQ)

**confusion\_matrix()** method from **sklearn.metrics** is used to get a confusion matrix in python.

****

|                                                                                         |
| --------------------------------------------------------------------------------------- |
| from sklearn.metrics import confusion\_matrix print(confusion\_matrix(y\_pred,y\_test)) |

Output - 

![](https://lh7-us.googleusercontent.com/XAayo6TWc5u4jhGvXzySQ7JZqkLOpJFDF7bWazrPpci44t83iRtruZlCk_zsRTmwYRR0rlyaJdJvRVkfuM15R4JaOLW79TUatgdkxsEL9Y9BZnJNpNVTNHEdaL-z0oYBpuGroKC7amRPjRM2DAcvtb4)

**2. Precision and Recall**

**Precision**

As the name suggests, it tells how precise(sure) we are about the predictions. In other words, out of total positive predictions how many are actually positive. 

![](https://lh7-us.googleusercontent.com/l0F5KbXytZnOiMNRx8BcFw3zgKXMYv_gVHL8smI51pmHdelyxTlOQPp7qDkyJBCzk7YoiOt3SWir-JS7RmzZ14C_do3o9M2xmcRVTs5FtOEX9GpbJ6KlYM32ak8xNjFGYgL-6pL7ZArrkLhuqZeKyJ4)

Remember the cancer detection scenario where the model predicted nothing as cancerous so precision = 0, whereas the accuracy is 99.90%.

**Recall**

It answers a different question, that is out of total actual positives how many our model predicted as positive. It is defined as

![](https://lh7-us.googleusercontent.com/Daz-3DIrFHMWTLEdzaGdKygkbW7gwgiNSk_ZkKz_zR8NnzlgXDgH6h6Z8ON6SgXxVVwdCKPaPRpnac3vBnW3buH-4b-fmtqUwEZlNTLglO1M3Gut5FjH-Al-iOZYcDIpwdp7LW7jE7pHm7JcdvLOVzg)

Similarly, for the cancer detection scenario recall = 0 as nothing is predicted as cancerous by the model.

Precision and recall both range from 0 to 1. Both are class-specific i.e. if there are 4 distinct output classes then there would be 4 precision and recall values.

It is always good to have a model with high precision and recall but generally, we don’t have the best of both. There exists a trade-off between precision and recall, therefore, we use F1 score that maintains a balance between precision and recall.

**3. F1-Score**

F1-score combines both precision and recall. It is a weighted average (harmonic mean especially) of both precision and recall.

![](https://lh7-us.googleusercontent.com/uyY1IHkeyBW0FDnDKkEPWgLvCBlSeIuWDaUVO0nqEQFz7m2Wh9mY4W9rvC4jrk_7drLYh2kf6SBvJfeEWK2eNKle0ihMYLYwTaNNNtLGzu-WdXtSD8B-djx8q-BAfpB6dTeTCyj214HXWjm-qjTzWQU)

Higher the F1 score, the better the model. Similarly, if precision is low then F1-score is low and also if a recall is low then F1-score is low.

Precision, recall and F1-score can be obtained together using the **classification\_report()** method from **sklearn.metrics** module. Let’s obtain these for the earlier built decision tree model on the news category dataset.  

****

|                                                                                                   |
| ------------------------------------------------------------------------------------------------- |
| from sklearn.metrics import classification\_report print(classification\_report(y\_test,y\_pred)) |

Output - 

![](https://lh7-us.googleusercontent.com/cUZQFieNXZjh_9DgOA_AtCRzEbfXDlQAeoz_gXM30etQq1cWic3g9CthBpFNHG8V9HVPmWziMP3Q8EU_ylm6jX8M3Hgufvc5bE3Y70HloTytm9o-6crvDEK0zRMfayIddfZi01Hrz-AXyLqefRLmS_I)

**7.5 Fitting various other Machine Learning algorithms**

Having seen decision trees, now let’s talk about logistic regression and SVM.

**1. Logistic Regression**

Logistic regression(LR) is one of the most commonly used classification algorithms due to its easy interpretability and fast computing. It predicts the probability for each class of the target variable. In other words, it provides the probability distribution of the target variable across different classes.

Considering the below news articles from the news category dataset, logistic regression may return a probability distribution across all four categories as shown below.

![](https://lh7-us.googleusercontent.com/JItjT2ul9JQws3yc52ksYi8vDaALVgJH0RnNGbAGR9ONspSNf2AoDzRe_eiV3lMMN2OCMeU22JfwTlsYMkbtp7nFKLYUXk1TMC83BmZoAQw-8AZLWDiYy5hanW_IhxbqPgPCcIOlpZL6hBqYlmCBsHs)

The above table can be interpreted as -

- For the 1st article, there is a 96% chance that the article belongs to the “World News” category, 2% chance that it belongs to the “Business News” category and so on.

- For the 2nd article, there is a 67% chance that the article belongs to the “Science - Technology News” category, 18% chance that it belongs to the “World News” category and so on.

- For the 3rd article, there is a 96% chance that the article belongs to the “Sports News” category, 3% chance that it belongs to the “World News” category and so on.

- For the 4th article, there is an 81% chance that the article belongs to the “Business News” category, an 18% chance that it belongs to the “Science - Technology News” category and so on.

Logistic regression is a special case of linear regression where rather than directly outputting the weighted sum of inputs, it passes the weighted sum through a function that returns values between 0 and 1 (probabilities). This mapping function is known as the **sigmoid** function.

Mathematically, linear regression equation - 

![](https://lh7-us.googleusercontent.com/o3Uz0LCjmoDzb-Kw2EFvFTKK1iVbSYtk0674To1RLQhis2BERHfJlVsDdRw-hNtfrUEDHct0MnOjrx5_e5RwjhqG6RptUKHmUevReCtzRADe5WRrGNhg22juGcxVjHIcU0GoEdj1s8K712sGYWnDvxs)

Sigmoid function equation - 

![](https://lh7-us.googleusercontent.com/yqiANkhAU0jMzK2k-etH-9_97eq0GoJZqiyNXi0s0pShYs43Cmx5KvpvoNQ4oQuZ67CLy9d9Fv9fdVr5KxZwK8euUs1caTvSv4t-Q-HuiANjwBcd0n2ZsA7saeKugtysuEqVtg0IrhM_z3o8faR41Cg)

After combining, the logistic regression equation becomes

![](https://lh7-us.googleusercontent.com/chKFVttfyxsNzvANUZ9dasMclA2qqp0ZfETlmzbtd6MEZQrvOj9RHzuu0DnOAIRDLpH8yF9MdUpa-4grgyuKCwV0y-SnQESJpags6JjztUn4ShQjx-5cRLSl7QAZoHwkpqpkDqu1t2fYq9Dn5v4w05s)

**Sigmoid function**  

It follows an “S” shaped curve as shown below.

![](https://lh7-us.googleusercontent.com/7XHhRJMy7xBIqwN5bikMEECvlxnZhQTQ9UroNyzgRIZVwUM-zo-AQ__ZV4k5jEJDpFYvohC8uEbzYgIx3Ae7A35VjgFQNiJiFTG_X40aWm4XXS4daUXcYxCovi_DrmWDg0DV-M_VO4DpvrfXYXYK09c)

Its output value always ranges from 0 to 1. For a large positive input value, it tends towards 1 whereas, for a large negative value, it tends toward 0. At x = 0, the output value = 0.5. 

So if the probability value is greater than 0.5, the observation is classified as a **positive** class (Y=1). Similarly, if the probability value is less than 0.5, the observation is classified as a **negative** class (Y=0).  

This threshold (0.5 here) can be selected based on the domain knowledge and type of problem at hand. Additionally, the hit and trial approach can be used.

**Implementing Logistic Regression in Python**

LR in python can be implemented using the **LogisticRegression** class from **sklearn.linear\_model** module. Using the **fit()** method, LR can be easily fitted on training data.

Let’s fit an LR model on the news category training dataset.

|                                                                                                                                                                                                |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| from sklearn. linear\_model import LogisticRegression lr\_classifier = LogisticRegression(random\_state=42) #fitting a Logistic Regression model print(lr\_classifier.fit(X\_train, y\_train)) |

****

Output - 

![](https://lh7-us.googleusercontent.com/02YJYS8pYBnUYiPQDFt7wUMVVGh6OdsU9jjzCVZUVbYMCDr8b16pAFdKtmvE0FnbxnY9k42njE5aduwVtiGgFaTFN1WM2hSe19qn_OrPfI1IdHBSXMAwpHUafz4lHMCoHJ28kNqsRDFGLRXULh6-91A)

**Making predictions and evaluating the performance**

|                                                                                                                                                                         |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| y\_pred = lr\_classifier.predict(X\_test) lr\_accuracy = accuracy\_score(y\_true=y\_test, y\_pred=y\_pred) print("Accuracy using Logistic Regression : ", lr\_accuracy) |

Output - 

![](https://lh7-us.googleusercontent.com/GTBxeUF2RwJYgkXYORp7k8e8ZlGs4yRTTqoBnuYhBWVfKZma2k1dVg0J1Guvp8lMTWhhF6C231bHKENx1IfmWGYYlZGFjcaw-i2SUDFUt3XN9qyTIw7X2kF_xJ2VseklIhTGfhUjmVz8WyAfspbOtB0)

This LR model is 84.93% accurate on the news category dataset. Similarly, the classification report is as follows

|                                                |
| ---------------------------------------------- |
| print(classification\_report(y\_test,y\_pred)) |

&#x20;![](https://lh7-us.googleusercontent.com/ueKLfDx7i8HSvJlcd5FRxG3VHqbDcy0rr8x3Axo8J23laNCLGLMI4erXKZsRmmPhkgMLEcPrG8BztXk1MxDRIUo7XUq9ke0gubQ12ozrgoDAc1wO_DHhpcA__xysmc4McdRaAwau6JT1IEi2y1mZBtk)

Precision and recall obtained using the LR model is far better than the decision tree model, similarly, the F1-score(0.85) is also better than the DT model.

**2. Support Vector Machine (SVM)**

Support Vector Machine (SVM) tries to find out an optimal **hyperplane** in a **d**-dimensional space (where d = number of features) such that it can easily separate the data points into classes. 

A hyperplane is a decision boundary that helps in classifying the data points. Its dimension varies as per the number of features in the dataset. E.g. For a dataset with two features, the hyperplane will be a line whereas for a dataset having 3 features, the hyperplane will be a plane in 3d space.

Consider a 2-dimensional hyperplane (basically a line) as shown below. Any data point that falls to its left, is classified as green. Similarly, anything that falls to its right, is classified as red.

![](https://lh7-us.googleusercontent.com/qKyAcoj3VcnZtK7za3J95THA4RHMGWtJ4fr04Aja0y5gHlZh4diB24ZtWprDWkrcUMATOGIWOpsqxHIGai9Y2DS1c_CQTL3ord7wjUFuXZzaMNA2_1Z_8keNw1OldFbNS1yvR_ixC_x_1t3FpPbe2WU)  

![](https://lh7-us.googleusercontent.com/EWrT60wgaNX7oihIpf6jRPIbMfpUqD7PktUlLqNP3I48y2ge7wj756WtqWeXzDhs4WSuh6jLgyd2wY5xmy-t7kkISOafdqvPm7IDE1klMMsXdri8pzBhZfDRsgaPMUQFGUlt_TZc4uAk_ZP719eX-nM)

Multiple hyperplanes like A, B and C (as shown above) are possible then how to find the best hyperplane?

The distance between the nearest point of the classes and the hyperplane plays an important role in finding the right hyperplane. This distance is also known as **margin** and SVM attempts to find a hyperplane with the maximum margin. 

**Support vectors** - These are the data points(vectors) that are closest to the hyperplane and dominate in deciding the position and orientation of the hyperplane.

![](https://lh7-us.googleusercontent.com/PdXBmWouFbQE-LG1ceC36wPaVeu4QKrfJDAH03KSbjaTdeUnez3Vix4VEOCoxCPzspVEVdS162CZWSq3NKCEoOOF_yo5pNzCUuUnjwxlGheei4fskn2xXy9liIRgDVVRnumOfRbcSZgPeEUKWo7GQkE)

**SVM types**

SVM can be broadly classified into two categories.

- **Linear SVM** - Suitable for linearly separable data. If a dataset can be easily separated into two classes using a single line then it is known as linearly separable data.

- **Non-Linear SVM -** As the name suggests, it is used for non-linearly separable data. Here, the dataset is mapped to a higher dimensional space using a kernel trick to get an optimal hyperplane. The most commonly used kernels are polynomial and radial. 

Below are some visual examples of non-linearly separable data.

![](https://lh7-us.googleusercontent.com/W9lgWb29rDMiRgzSsUziaaeBvLiSBk2vIHeFO-QTGNPn2r8mpyTbzTPtsQeKzyu47G-I8UwcDpV5GAa0f2MIOmL4rDXZeVfOrNmSLeKsscgN386UubZ1r49vnG_gTqZR39zz-cbOWPtzBcHUtPBZe_k) 

****

**Implementing SVM in Python** 

The **SVC** class from the **sklearn.svm** module is used for implementing SVM in python. Let’s fit SVM on the news category training dataset by setting the kernel parameter to “linear” for linear SVM.

|                                                                                                                                                     |
| --------------------------------------------------------------------------------------------------------------------------------------------------- |
| from sklearn.svm import SVClinear\_svm = SVC(kernel = "linear", random\_state = 42) #fitting a SVM model print(linear\_svm.fit(X\_train, y\_train)) |

Output - 

![](https://lh7-us.googleusercontent.com/jFYHuViG17r0VDRmAAjPNL0p03qDXhyIFoBwZKVvcNn62f9q09u0UcksXe_iwOvKpSc-Qn8EdeOICpteogjapP52_FLeS_81FZ-qXmipds83A6RYrBUSr6ey3Pddd9SIuL3vEjiXvIttnbQoRVA3yNI)

**Making predictions and evaluating the performance**

|                                                                                                                                                              |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| y\_pred = linear\_svm.predict(X\_test) svm\_accuracy = accuracy\_score(y\_true=y\_test, y\_pred=y\_pred) print("Accuracy using Linear SVM :", svm\_accuracy) |

Output - 

![](https://lh7-us.googleusercontent.com/XF1RDyBMMrFShB64THgN6EMqcMVM-uO1b4rAzzHo4Bh9wstNjbVqyj8PNokxc12g_GH5N_XBsbAicjMDSVnQEM-Hm4eLqxVYX5OuzocnAf8XDxbx_nCqLMzHGOqNIYCWp5u5YIpSCehxgbycoapEwUQ)

So, this SVM model is slightly better than the earlier fitted LR model. Similarly, the classification report is shown below.

|                                                      |
| ---------------------------------------------------- |
| print(classification\_report(y\_test,y\_pred))       |

![](https://lh7-us.googleusercontent.com/dHhUy5NTTib1HxeqEg8zdOKUh7lfTBrmOebikxvEKZVynEZVK4nsK_mxmo4Dkioxq9QzJAO9uwHiJ8By-68xpQBs8xWFxEdgwogCe8_o08GZeG2fipl7F2jLNzr2e-mzCDbiOhAzaGLb3L4qbbVQADw)

**7.6 Persisting a model** 

Having seen the model building and evaluation part, now let’s talk about model persistence. On significantly larger datasets, model training may consume a reasonable amount of time (even upto some months). So it is not desirable to train a model again and again with the same parameters. 

Hence, persisting(saving) the trained model is helpful in its future usage without having the need to retrain. 

**Pickle** and **Joblib** are the two most commonly used libraries for model persistence in Python. Joblib is more efficient and fast on large numpy arrays over Pickle. 

Joblib has two useful functions (dump and load) for model persistence(serialization) and reloading(deserializing) a persisted model. Use _pip install joblib_ for its installation.

- **dump() -** To persist a random python object in a file. 

- **load()** -  To load a python object from an already persisted file.

|                                                                                                                                                                         |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| import joblib #Persisting the model in a file joblib.dump(linear\_svm, "linear\_svm.joblib") #loading a persisted model linear\_svm = joblib.load("linear\_svm.joblib") |
