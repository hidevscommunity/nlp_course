**Chapter 4. Approaching NLP problems** 

Having known NLP basic terminologies, now let’s see how to approach NLP problems. Generally, while dealing with an NLP problem, we break the problem into subproblems and then follow a step-by-step procedure to solve them. These steps can be integrated together into a pipeline, known as the **NLP pipeline**.

**NLP Pipeline**

Let’s talk about a generic NLP pipeline and its components as shown below.

![](https://lh7-us.googleusercontent.com/0Gtfcsu0cA6frdynFZckjyzfhz5uGApJ_7iquhrF3B6zqla8NFNm7AKAz3THW6vE-oi5jiiAH5ajBb4fFpXResakTzcURHZk08lBCN5wTovFRRdjQt66KeKCgOQAb5f2DAzxHTtcblmTU5j-Vab1Ppg)The main components of this pipeline are - 

- **Basic Data Analysis**: To get basic and useful information about the text data, like a total number of observations, distinct values, their frequency & distribution, association b/w variables etc. 

- **Text preprocessing**: Cleaning and transforming text data to a more digestible and useful form for further proceedings. Majorly it involves lower case conversion, removal of non-alphanumeric characters, links/URLs, stop words removal, stemming and lemmatization etc.

- **Text Featurization**: Converting text data to numeric features so that they are easily ingestible to machine learning algorithms and help in learning better. Most commonly used featurization techniques are Bag of Words(Bow), TF-IDF, and word embeddings. 

- **Model Fitting & Evaluation**: Building multiple machine learning models on the featured text data and evaluating their performance to assess their usefulness/goodness. Selection of ML algorithms depends on the nature of the problem at hand, business requirements, domain knowledge etc. 

- **Model Deployment**: Making the fine-tuned model available to the end-users/client through model deployment. Once the best fit model is picked out, we proceed towards its deployment in production.

**4.1 Scenario-specific Case-Study - News Category Classification**

Plenty of news articles are published daily on various topics/events. Classifying them into an appropriate category is a challenging task as manual classification is not so feasible. In this case, using NLP and Machine Learning we will automatically classify the news articles into one of the predefined categories based on the article content.

![](https://lh7-us.googleusercontent.com/DU4ISDRAJzlREgb9RBqKyKrpmviHBfjj8Uiitob6Wf830ycKlBlZU9SBC4O3LoJ7VoKny2ZVQWpIulLuKn_pjJgcPnBgVMMdI717ohL-XMk9wh2IyQ0QutUri2C-Dbvd-sGUQk4H-dpTaNwm3j05aJA)

So let’s get started by using the **News Category Dataset**. The dataset contains around 14k observations of 3 different features (“Title”, “Description”, “Category”). Here, the distinct news categories are “Business News”, “Science-Technology News”, “Sports News”, and “World News”.

**Tip** - The above dataset can be downloaded from [here](https://drive.google.com/file/d/1EDNZ7ruyVQGdCeX4yIKTx5lYLq51tIJo/view?usp=share_link). ****

We will approach this case study as per the above-discussed NLP pipeline. All the components of this pipeline will be discussed/explored in a detailed manner. This generic pipeline can be used for similar kinds of NLP tasks.  

****

**4.2 Basic Data Analysis**

Let’s perform some basic data analysis on this dataset.

**NOTE** - For the hands-on purpose, we are using **Jupyter Notebook** IDE throughout the course. Jupyter Notebook is an open-source web-based interactive application, ideally designed for data science and machine learning workloads. It allows us to write and share the live code, equations, visualizations etc. The highly recommended way to install the Jupyter notebook is by downloading **Anaconda**.

**a. Loading the data**

For data loading, preprocessing and manipulation-related tasks, the **pandas** python library is being used here.

****

|                                                                                                                                  |
| -------------------------------------------------------------------------------------------------------------------------------- |
| import numpy as np import pandas as pd #loading the csv file in a pandas dataframe news\_df = pd.read\_csv("news\_category.csv") |

****

**b. Basic statistics - Number of articles, categories** 

|                                                                                                                                                                                                           |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| print("Total number of articles : ", news\_df.shape\[0]) print("Total number of unique categories : ", news\_df\["Category"].nunique()) print("Unique categories are : ", news\_df\["Category"].unique()) |

Output - 

![](https://lh7-us.googleusercontent.com/avYqdF_LftsgV-jwUviHK1JU5-FSFTaCOXK--inc8n_wXVlgO9yZ_43prKX6X-lgZ6bRYqcoH8RpeKnbFVH7_I1ghtnzUcCyEXoBO4YWRQDhlNU1kIAQUOitOO39BgiFq3GK0uakZaolMjAe0bM_Cg8)

So, the dataset contains 14400 observations across 4 different news categories.

****

**c. Frequency of each category** 

 news\_df\["Category"].value\_counts()

****

Output - 

****

![](https://lh7-us.googleusercontent.com/4fNm3khkocxR2jozeUH_TOmV9ecJ-6T6_lVg6bNV9-ohOuXQwXvzpCzrDFaD8wsFA5EcJcd7RQy6eQkFZXBfvtLeIZFEyDPqb6DqSvTY66F-3IFeI0qNWeJaNVJvQXpNJbSO9dKr3En6Q56JV1JeLXI)

****

As we can observe the data is almost balanced. For better analysis, let’s plot the distribution of articles.

****

**d. Distribution**

 ****

****

|                                                                                                                                                                                                                                                                                                              |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| import matplotlib.pyplot as plt import seaborn as sns plot = sns.countplot(news\_df\["Category"]) plt.title("Distribution of articles category-wise") plt.xlabel("Category name") plt.ylabel("Number of articles") plot.set\_xticklabels(plot.get\_xticklabels(), rotation = 45,horizontalalignment='right') |

****

Output - 

****

![](https://lh7-us.googleusercontent.com/wsLlqQFvGTo8x-TLzNnbqFpFsn4hp_jp8o9hMOYKd5PI-V71fVJdea2t2CLBWqDPLh92sze9ReOgW30yO9tx65vEefjvhqWIF1zoJ3mHRK3iMGtXM3lz0n_o2cXWEraXehWwe_ltcixu8tgv0lHhDZo)

From the bar chart, it is clear that the “Science-Technology News” category has the highest number of observations then “Sports News” and so on.
