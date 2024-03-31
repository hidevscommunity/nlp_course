**Chapter 2 - Getting Started with NLTK(Natural Language TookKit)**

****

There are various NLP libraries in Python like NLTK, spaCy, Stanford CoreNLP, Gensim etc. to analyze, process and understand natural languages.

 

**NLTK** (Natural Language ToolKit) is the most dominating and easy-to-use suite of libraries for NLP workloads. It has over 50 corpora and lexical resources such as WordNet. 

****

Using NLTK, the following tasks can be easily performed.

- Tokenization

- Stemming

- Lemmatization

- Parsing

- Tagging

- Semantic reasoning

- Classification

****

**Installing NLTK and its components**

****

Use the below command in the terminal to install NLTK.

****

|                  |
| ---------------- |
| pip install NLTK |

****

In Python shell or Jupyter Notebook type “**_import nltk_**” if it does not show any error that means successfully got installed.

Similarly, use the following commands to download NLTK components.

|                             |
| --------------------------- |
| import nltk nltk.download() |

****

A GUI pops up as shown below

![](https://lh7-us.googleusercontent.com/UtDlexkintf9J4rwDEAVFrw7P4bXcDQDkzDUoOjib3XuYLbiCZzOQNbxXy2SydBVYixEgR3_QRkd4TqDPB5ZjBkjPpuQvQz1cd4a3U4GbUsYBwiDbsK_dRdd3a4vvsadhSyb-rXhM8BaS4g6s3ujz7s)

****

Select “**all**” as highlighted to download all the packages. It covers tokenizers, chunkers, all corpora and other algorithms. Downloading will take time-based on the internet speed. Here, you can also change the “_Download Directory_” by clicking on the “_File_” tab.

**NOTE** -  Instead of downloading all, you can also download the required things manually. For e.g. To download a list of stop words, use - 

|                            |
| -------------------------- |
| nltk.download('stopwords') |
