**Chapter 6 - Feature Engineering (Featurization) of text data**

Numeric features are easily ingested by the machine learning algorithms and a model’s performance is directly influenced by the features used. But many times in various scenarios, the data is not directly available in numeric format, e.g. here the **news** title and description are in text format. 

So we need to convert the text data into numeric features and this conversion is called featurization of text data. 

**Why do we need to featurize the text data?**

To convert the text data into numeric features so that ML algorithms can easily explore, analyse, compute and learn from those numeric features and perform the relevant task better.

There are multiple ways to represent the text data as a **d**-dimensional vector(numeric feature), the most commonly used are - 

1. Bag of Words

2. TF-IDF

3. Word embeddings

**6.1 Bag of Words(BoW)**

This is the simplest and effective approach for text data representation. It represents a document(row) as a numeric vector by counting how frequently each word appears in the document. 

Consider the below sentences.

|                                                                                                                                         |
| --------------------------------------------------------------------------------------------------------------------------------------- |
| "This course is very interesting but lengthy" "This course is not much interesting but short" "This course is interesting and is short" |

****

**Step A. Define the vocabulary** 

Vocabulary is defined as the set of unique words in the document set(corpus). The size of the vocabulary determines the dimension(d) of the numeric vector. 

Here the vocabulary from the above 3 sentences is {“This”, “course”, “is”, “very”, “interesting”, “but”, “lengthy”, “not”, “much”, “short”, “and”} and dimension d = 11.

**Step B. Calculate the frequency**

To featurize a document(sentence), calculate the frequency of each word in that document. 

So, the 11-dimensional sparse vector for each sentence looks like - 

![](https://lh7-us.googleusercontent.com/6G7qZ8Sac0_o-WF5VS8762CgL-ksjQg2TkWgflySyTB8nGfvGmCjbEp2dqC8Az_FkPLFf2-uGYtfgiuYXBUcU35Opgg0tMP1J7mJ-Ek7YSYV9x7LtiNbqckMkz6mGakqLPQ7RxduU_5s7kQ0sDsq_6U)

This above table is also known as the “**Document-term matrix**”. Here, the vectors are - 

![](https://lh7-us.googleusercontent.com/xw4eGfBauXt5sTOAVQxg8T-fTzzWzs-y7iOWt1B6PX5913kpLVj3QGU6NNk__gUcmT2GnB5cc9ZmfOPGo6Ze6SCw3vXpzEoCMI9b2AM-pc_jMdBtYKeJIhFEUIueD_xJ9WYPb6GHENi93dK6Wiwzc0k)

****

**Python implementation of BoW**

In Python, using _CountVectorizer_ from _sklearn.feature\_extraction.text_ module, BoW based featurization can be easily done.

|                                                                                                                                                         |
| ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| corpus = \[ "This course is very interesting but lengthy", "This course is not much interesting but short", "This course is interesting and is short" ] |

****

|                                                                                                                                                                                                                                                                                                                                                                                                |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| from sklearn.feature\_extraction.text import CountVectorizer vectorizer = CountVectorizer() extracted\_features = vectorizer.fit\_transform(corpus) #featurized data vocabulary = vectorizer.get\_feature\_names() print("Vocabulary : ", vectorizer.get\_feature\_names()) # to get unique words # for better visualization pd.DataFrame(extracted\_features.toarray(), columns = vocabulary) |

Output - 

![](https://lh7-us.googleusercontent.com/hu_dlF5lmkzwTuu0A6V7Ln4wOfVZRX5O-zjHm55DGjeTLGbFCx5d7F7aUb7qryLGzGaSFYNunHDByl5t8qMCi5dRB3QpPP_RwdFh61DxI7zKVBz3AAVCkGG2GBW6tDzA7SLLeVdDbYveSI2bWeSUktY)**Code Explanation**

- fit\_transform() - determines the vocabulary based on the data passed and returns the document-term matrix.

- get\_features\_names() - returns vocabulary i.e unique words from the corpus.

**Disadvantages of BoW**

1. It does not capture the sequence/order of words. For e.g. the vector for “this interesting course very is lengthy but” would be similar to sentence 1.

2. Also, it does not capture semantic relationships(words with similar intent).

3. It generates a very dense sparse vector for larger corpus(set of documents).

**6.2 TF-IDF(Term Frequency- Inverse Document Frequency)**

TF-IDF is a weighted measure that tells how important a word is to a document or corpus. As the name suggests, it is a combination of two matrices.

1. Term Frequency

2. Inverse Document Frequency

Collectively, it is defined as TF-IDF = TF x IDF

**a. Term Frequency**

Term Frequency(TF) measures how frequently a term(t) appears in a document(d). Mathematically, it is defined as

![](https://lh7-us.googleusercontent.com/s0Yq1ZLT3xFGX6YX-9-z6I3v2225tj84n8ewcev8F8-ySURXPgxuWI00t5dyYWCo1ISlvciomMmr-GdcfTRDPu7ai1sdrtPFkEhz1pLMwIh1k04pipbe2oC202eA240aHuiTrh_X4WHqDt-7t6Wel7w)

Let’s compute TF for sentence 3 with the same vocabulary. 

Recall, sentence 3 = "This course is interesting and is short"

Vocabulary = {“This”, “course”, “is”, “very”, “interesting”, “but”, “lengthy”, “not”, “much”, “short”, “and”}

****

TF(“This”) = TF(“course”) = TF(“interesting”)  = TF(“and”) = TF(“short”) =1/7 = 0.142

TF(“is”) = 2/7 = 0.285

TF(“very”) = TF(“but”) = TF(“lengthy”) = TF(“not”) = TF(“much”) = 0/7 = 0

****

**b. IDF (Inverse Document Frequency)** 

IDF is used to measure how important a term is. It is defined as -

![](https://lh7-us.googleusercontent.com/Hs0DiodQq1E_veR3yfzQVVpg3nbgV55q24r5ku5UwSbYloHlUbJi-f5QnJwbtuYIRCS4NK9T2owtt7MQIlm-p3FWlc20ESexINxHj5BJ0Z8nMbYd1mvwVflqdgvhomIrxLPwqBoCiPLfSYGAhSUUO-U)

****

Similarly, IDF for terms is -

IDF(“this”) = IDF(“course”) = IDF(“is”) = IDF(“interesting”) = log(3/3) = 0

IDF(“lengthy”) = IDF(“not”) = IDF(“much”)  = IDF(“very”) = IDF(“and”) = log(3/1) = 0.48 

IDF(“short”) = IDF(“but”) = log(3/2) = 0.18 

This implies that terms “course”, “interesting” etc. are less important whereas “lengthy”, “short”, “very”, “much” etc. are more important in this corpus.

Further, TF-IDF can be computed as 

![](https://lh7-us.googleusercontent.com/0RT8TfV33aEAlumC2AjeUj2YBzULYFWV_EEah9xUGaLuJISSinfl3hzFv48_Te6ulriWCuMXo1OjMA_lrVTm8HOhAsPNp9RE-NTaozMJ5eOVkU0WGcN9IaC3Rr-zfmnuOwFz3yrR6EZvU510NbjL-Xg)

TF-IDF score will be high if both TF and IDF values are high. So if a term is frequent in a document but rare in the corpus then its TF-IDF score will be high.

**Why use TF-IDF?**

It gives better significance to less frequently occurring words in the corpus than Bag of Words(BoW) based representation.

**Implementing TF-IDF in Python**

TF-IDF can be implemented using _TfidfVectorizer_ from _sklearn.feature\_extraction.text_ module.

|                                                                                                                                                                                                                                                                                                                                                                                                                            |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| from sklearn.feature\_extraction.text import TfidfVectorizer tfidf\_vectorizer = TfidfVectorizer() extracted\_features = tfidf\_vectorizer.fit\_transform(corpus) #featurized data vocabulary = tfidf\_vectorizer.get\_feature\_names() print("Vocabulary : ", tfidf\_vectorizer.get\_feature\_names()) # to get unique words # for better visualization pd.DataFrame(extracted\_features.toarray(), columns = vocabulary) |

Output - 

![](https://lh7-us.googleusercontent.com/UelVd5Uy3aFglgAmau6pJHvQ8-3C_Wg3AsnIql4FYt1_I1ahWj_8ojhiLzD6uRVg70HX00rRv8IssnwNnQl5fkgZYwJHBXRWwfCM0Acm3JGwWOweUxkZQULQ3vSDcpCKRAf0-6dlVSlBiRwAg0R6Nag)

 **Disadvantages** 

- Like BoW, TF-IDF also does not capture semantic relationships among words.

**6.3 Word Embeddings**

Remember, we specified Semantic similarity as a drawback of BoW and TF-IDF. It measures how similar the words are. For e.g. there is a good relationship between the words like “father” and “mother”, “panther” and “tiger”, “india” and “delhi”, “school” and “students” etc.  Now, how to capture this semantic relationship?

Word embeddings allow doing so. It embeds the words into a feature space such that similar words are clustered together in their proximity. Consider the below mappings.

![](https://lh7-us.googleusercontent.com/UibRu_UI6Z1TWqxQrMaLYXuKKlyvHhBh7b2qg-U9FY6gp1yfKrO-GyMTN4jE1BW_tN3W7njGQDS6ttOt7J_oeDxBmGY1NoQIppQDogPoGxNLn5CHIQnImEkR6M0TCDKCYRiL_PTsODHot8EpY0m3v-g)

Here, we are talking about 3 kinds of semantic relationships. The relationship between Man-Woman is similar to Father-Mother, between learning-learned is similar to studying-studied, similarly between India-Delhi is similar to Germany-Berlin. 

Using word embedding such mapping can be easily done. 

**Why use word embeddings?**

- To capture semantic relationships among the words.

- To get a dense feature representation of text data as opposed to sparse representation from BoW and TF-IDF. 

There are multiple word embedding techniques like _Word2Vec_, _GloVe_, and _fastText_. Let’s discuss _Word2Vec._

**Word2Vec**

Word2Vec was invented by _Google_ in 2013 with an objective to generate vector representations of words in such a way that words with similar intent are clustered together. Such vector representations can be further utilized for various NLP tasks. 

It gives better results on quite a large corpus. Since our news category corpus is not too huge, it is better to use Google’s pre-trained Word2Vec model. It is pre-trained on millions of google news articles and provides vector representation for 3M words and phrases where each is represented by a 300-dimensional vector.   

**Implementing Word2Vec in Python**

To download and load this pre-trained Word2Vec model, we are using the _gensim_ library. (use _pip install gensim_ to download this library). Gensim also provides functions to compute similarity, finding the most similar words to a given word etc. 

|                                                                                                                                                                                                               |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| import gensim.downloader #downloading and loading the pre-trained model model = gensim.downloader.load("word2vec-google-news-300") #finding most similar words to a given word model.most\_similar("panther") |

Output - 

![](https://lh7-us.googleusercontent.com/T7m9gbfBX9JEBpiYTK9oD7mytGaDAcLZFBIKWWylZxu2HCjZgwbzhY9K7hs36W78G4yT-7WL7fy2WVNE0vYWs7mQn55oLprc3GkTMX88zLoLmzzrLbnpvY9BVvu_UTNq7oPWFeXiIx8xUAEUm0h3b5A)

As we know that “panther”, “tiger”, “cougar”, “jaguar”, “leopard” are from the same family and it is easily figured out by the word2vec model.

Similarly, we can examine other semantic relationships as follows.

|                                                                                                                                                                                                                                                                                                |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| #checking similarity print("Similarity score b/w father and mother :", model.similarity("father", "mother")) print("Similarity score b/w clothes and garments :", model.similarity("clothes", "garments")) print("Similarity score b/w india and delhi :", model.similarity("india", "delhi")) |

Output - 

![](https://lh7-us.googleusercontent.com/ltC4wSFUIee4DKvDSbAudES-RMqE7lic7fW8OhqJhFEwsBqI-sFMTjIuR0EYjJV-L6YBfJPW4T81YCsujqM7d_z5HeMwTBDMKMviGVHBTxdX5N2yvMr3OvGvX5_WsHIgCxSeX8tMXg4ZKwFcuO3CgKA)

****

Let's check the vocabulary size and vector size of this model.

|                                                                                                                 |
| --------------------------------------------------------------------------------------------------------------- |
| print("Vocabulary size of pre-trained model :", len(model.wv.vocab)) print("Vector size :", model.vector\_size) |

Output - 

![](https://lh7-us.googleusercontent.com/QZ3K3jpbeuu8UYCJU60VJv4FyGhBX2kxru4sjYHEQN1dHeRA6iG7ACBvne9e21Mli5gP77RbWnt7LLMGuptWoK952NiEn7-sMLYMdReoljTK2Z5CBkNaIhXNlDOo2-G-_nC9tC7Yj1VFVtirWke-obE) 

**Featurizing news articles dataset using Word2Vec** 

Since this pre-trained model is 3.6 GB in size, loading this in RAM may take significant time in the final deployed applications, therefore, let’s use a lighter version (58 MB in size) of the same. Download it from [here](https://drive.google.com/file/d/13d5i0Jo9Uh7MHJsrk6Ax9UM_BamE9sat/view?usp=share_link) and load it as shown below.

**NOTE** - It is always recommended to continue with the full pre-trained word2vec model, here due to memory constraints we are proceeding with a lighter version of the same.

|                                                                                                 |
| ----------------------------------------------------------------------------------------------- |
| import pickle with open('word2vec\_model', 'rb') as file:     loaded\_model = pickle.load(file) |

Word2Vec model provides vector representation of a word but here we need to featurize the entire news article summary. The easiest way to get this is by computing the simple average of all the word vectorization within an article summary.    

|                                                                                                                                                                                                                                                                                                                                                                                                              |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| vocabulary = loaded\_model.keys() w2v\_summary = \[] for i in news\_df\['news\_summary']:     w2Vec\_word = np.zeros(300, dtype="float32")     for word in i.split():         if word in vocabulary:             w2Vec\_word = np.add(w2Vec\_word, loaded\_model\[word])     w2Vec\_word = np.divide(w2Vec\_word, len(i.split()))     w2v\_summary.append(w2Vec\_word) w2v\_summary = np.array(w2v\_summary) |

This code performs vectorization of all the news summaries from the news category dataset. 
