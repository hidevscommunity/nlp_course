**Chapter 5 - Text Pre-processing** 

**Why do we need to preprocess the text?**

Since the writing(wording) style may vary from human to human so it is not recommended to use the raw text directly into machine learning models. For e.g. “ten” may be written as “10”, “TEN”, ”Ten”, “ten” etc. Similarly, different forms of verbs indicate the same action(intent). For e.g. “study”, “studies”, “studying”, “studied” have the common action “study”.

Machines are not as great as humans to interpret such variations in the writing styles. Therefore, we need to standardize such words using Text Pre-processing.

In this section, we will discuss the most commonly used text pre-processing techniques. 

**5.1 Basic text pre-processing**

**a. Lowercase conversion**

Converting the input text to the same casing(especially lower case) guarantees similar treatment of the words, for e.g. “Pre”, “pre”, “PRE” will be treated as “pre”. Also, lower casing reduces the size of vocabulary.

|                                                                                                                                                       |
| ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| text = "Dual-core   chips that AMD and Intel plan to bring   to market next year won't be sharing their  memories\*." text = text.lower() print(text) |

Output

![](https://lh7-us.googleusercontent.com/tDhVr4d5JNhneqUXJrhXKoqRzDxxWv-37RnmhUxyuMn93jXuGm3SWptlY4bI7eFwMkVEW6eNWYDmNnXmtwT63plL2muxCfDuBliHNCsT3dwNtXy-4ni9c6iJCQrVjmSnG8_3cKKw29oUAt8-0rf4VTQ)

**b. Removal of extra spaces**

We can easily substitute extra spaces by a single space using **sub()** function(from **re** module) with the help of regular expressions.

|                                                      |
| ---------------------------------------------------- |
| import re text = re.sub("\s+"," ", text) print(text) |

&#x20;Output 

![](https://lh7-us.googleusercontent.com/xCSyK4c4Zms-RdnEjzzwbJiQvv4JM3-7qE6G_xxuZG8P_lJgMMs2B8CZVjHdu5Ya2iUznCGGGQVf9sLkdKVp32rzNPLR2RTzAZR8PyNesz2ZMRoEBk_avwa6m7vdiIAbrTNth3nU5uUtVMYL6gCOZH8)

**c. Text decontraction(expanding abbreviations)**

“won’t” and “will not” have the same meaning but machines may interpret them differently so it is better to decontract such abbreviations.

|                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| def decontract(text):     text = re.sub(r"won\\'t", "will not", text)     text = re.sub(r"can\\'t", "can not", text)     text = re.sub(r"n\\'t", " not", text)     text = re.sub(r"\\'re", " are", text)     text = re.sub(r"\\'s", " is", text)     text = re.sub(r"\\'d", " would", text)     text = re.sub(r"\\'ll", " will", text)     text = re.sub(r"\\'ve", " have", text)     text = re.sub(r"\\'m", " am", text)     text = re.sub(r"\\'m", " am", text)     return text text = decontract(text) print(text) |

Output 

![](https://lh7-us.googleusercontent.com/pwP6nOoFZuJt0vQlaQSOM0kzlrsP7WscpkQ1aHibyqGToT1iuboESKnT6Qe_wjnrFrZI1zcqBDOdBbxx4ZV2URvlS8CPedYmXqDAqEV7IjDJapktPwnDUuj5K8pCA3bi6cJlQNC2vbB1tlQzLtll_wQ)Here, “won’t” is decontracted as “will not” using the decontract**()** function.

**d. Removal of Non-alphanumeric characters**

For analysis, it is preferred to have only the words with alphanumeric characters(a-z and 0-9). Non-alphanumeric characters can be identified using **\W**. Here, we are replacing them with a space.

|                                           |
| ----------------------------------------- |
| text = re.sub("\W"," ", text) print(text) |

Output

![](https://lh7-us.googleusercontent.com/H7iKj7p-nA4uKxBAnaGyjmoADJDGT_qvcuHRq1vcFaJKsrIN-u0_9HuZIwLdQ5CbJoJyqbqsqdPxGABEKOoVK_wm7_r-LubjKJjcBTMhCL7kd4MqCMBHka1U0ukxqi69co_443BdX5TG47m5Ge2VI3A)

**e. Removal of links/URLs**

Some news articles may contain URLs in their description so we might need to remove them for better analysis.

|                                                                                                                               |
| ----------------------------------------------------------------------------------------------------------------------------- |
| text = "Sysco Profit Rises http\://www\.investor.reuters.com/FullQuote.aspx?ticker=SYY.N" print(re.sub(r"http\S+", "", text)) |

Output

![](https://lh7-us.googleusercontent.com/DeSc0BFXWLLGGgxy0MUatXv3-DTSMDFcx5bJgQxm21cWo2JBouLoafZ1pBBCSqzWg0y_MyTrDI-6_YmzyE4wGMC0Pd3zhb6UcUsROsN-uq82gwpHOjnwMLr2R8BJlWGnP-eJ71OUJ59Co6b6ITpxSuo)

**Basic text pre-processing on news articles dataset**

Since the complete essence of an article relies both in “Title” and “Description” columns, let's combine them together as “news\_summary”.

|                                                                                  |
| -------------------------------------------------------------------------------- |
| news\_df\["news\_summary"] = news\_df\["Title"] + " " + news\_df\["Description"] |

Let’s apply all the above techniques on the new articles dataset. Here, **the preprocess\_text()** function contains these previously discussed pre-processing techniques.

|                                                                                                                                                                                                                                                                                                           |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| def preprocess\_text(text):     text = re.sub(r"http\S+", "", text)     text = decontract(text)     text = re.sub("\W"," ", text)     text = re.sub("\s+"," ", text)     text = text.lower()     return text news\_df\['news\_summary'] = news\_df\['news\_summary'].apply(lambda x: preprocess\_text(x)) |

**5.2 Removal of Stop words**

Stopwords are the most commonly used words in a language like “the”, “a”, “this”, “that”, “as” etc. Such words are not very helpful in analysis as they do not contain valuable information and have less prediction power. Also, their inclusion affects the processing time so we usually remove them. 

NLTK provides a pre-defined list of 179 English stop words, available inside the **nltk.corpus.stopwords** module. 

****

|                                                                                                                                                                                       |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| from nltk.corpus import stopwords stop\_words = set(stopwords.words('english')) print("Total number of Stop words: ", len(stop\_words)) print("\nList of Stop words : ", stop\_words) |

Output

![](https://lh7-us.googleusercontent.com/iXFJtYiMNAhjY9CI60plcIFGk2oA54hVeFuKrKPAcL91IVswNrBwq7ixiDyAOsTdFKen9cAIuxEQMuoZvpbwaVGAzIj31B3PH7I4iuyv53iy5yc9E_uRymD0Kb4jF1qHDvmFs0BjxLeRVN7Ia3vO23A)

Now let’s remove the stop words from the “news\_summary” column of the news category dataset.

****

|                                                                                                                                                                                |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| for article in news\_df\['news\_summary']:     article = \[word for word in article.split(" ") if not word in stop\_words]     article = " ".join(article)     article.strip() |

****

**5.3 Stemming**

Generally, in languages words are derived from one another, e.g. “walk”, “walks”, “walking” and “walked” are derived from the word “walk”.  Such changes in the form of a word are done to express grammatical distinctions such as tense(walk, walked), the person(I, we), number(dog, dogs), case(boy, boy’s)  etc. This process is known as **Inflection**.

An inflected word(s) belongs to a common root form. Let’s see some more examples.

|                                                                                                                                |
| ------------------------------------------------------------------------------------------------------------------------------ |
| "I am learning NLP crash course" "I learned NLP crash course" "I have to learn NLP crash course" "She learns NLP crash course" |

Here, in the above sentences - 

![](https://lh7-us.googleusercontent.com/qiFx6XnuymgxSg9TYo-ygI3_9sF4L6O_Z4mM9XJV8A2Llxl0Og3m2yrBcL3Avv6pHsJwfKrGYpOyf77HWS8TliL_ZkZPFXKi2QY18i5qOqfko-anGdPn-tnYoxClfBj9Dl5lApYBAqK3PpVPxuz45Zs)

****

**Why do we need Stemming and Lemmatization?** 

So it becomes essential to map such inflected words to the common root form in order to make a more intelligent and accurate machine learning model. This mapping can be easily done using **Stemming & Lemmatization**.

**Stemming** is the process of mapping a group of words to their stem(root). It has many applications in information retrieval, web search, SEO etc. 

Using **PorterStemmer** (available inside **nltk.stem** module), stemming can be easily performed.

|                                                                                                                                                                                  |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| from nltk.stem import PorterStemmer stemmer = PorterStemmer() words = \["learning", "learned", "learn", "learns"] for word in words:     print(word, "--->", stemmer.stem(word)) |

Output 

![](https://lh7-us.googleusercontent.com/ptvK4F5E6XKPoER13GtWu5O4o5pM2KRXR5eNG7SccCaoBqXqtM6D0Zoz-J6TqW9upKQdvWZC8hCho-OKemyCecdln89RFIVHNLAXiKaIp_zssq1D-O6GmmtIDQzcYDXQAaAGBeyxfS4fkz33hGdzfbM)

**Downside of Stemming**

Stemming is done by just removing suffixes/prefixes from the inflected words, therefore it may result in the stems which are not valid English words as per the dictionary. For e.g. consider the below words.

|                                                                                                                       |
| --------------------------------------------------------------------------------------------------------------------- |
| words = \["studying", "studied", "course", "explored"] for word in words:     print(word, "--->", stemmer.stem(word)) |

Output 

![](https://lh7-us.googleusercontent.com/eUnNsaYJdyvZj2LX2CLS301b2lwqFfFv-zarNRYFAMsdrvGQ4iaDhEPk7kd1CZbLOAJlFdNcy6Wnu1Wjm6vAJOO5HF0SutN2zkiWVW0v8uA0ZknDaZwa8CW4fDiPiMJYCuxOIbBxkmUdk2jt7arGY94)

Here, as we can observe the stems “studi”, “cours” , “explor” are not valid English words. To overcome this downside, Lemmatization is used.

**5.4 Lemmatization**

Like Stemming, Lemmatization also returns the root(lemma) word, additionally it ensures that the lemma(root word) belongs to the language. This makes lemmatization more powerful and informative. 

Lemmatization performs morphological analysis on the words by considering a language’s full vocabulary and the context. Therefore, lemmatization consumes more time in computation than Stemming.

Lemmatization can be done using **WordNetLemmatizer** from **nltk.stem** module. Also, we need to specify the context in which we want to lemmatize and this is specified using the _pos_ ****parameter. Here, it is set to “v” (verbs).

|                                                                                                                                                                                                                   |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| from nltk.stem import WordNetLemmatizer lemmatizer = WordNetLemmatizer() words = \["studying", "studied", "course", "explored"] for word in words:     print(word, "--->", lemmatizer.lemmatize(word, pos = "v")) |

Output

![](https://lh7-us.googleusercontent.com/ZL6-wAHboIl31IQ6ugSFf_t4PlJo_KKHPKb9Elqs409W0YDgJtl2gomK8hxlAaJrX2J2VUhYjC1X9dnDTv8Hm66D2ZKiFFeocyyqmY15FjwgRE3QXEF9iXzdsQGJ2_QZqBsbdX0eL_G0fgD9r4x9GBU)

Now, let’s apply Lemmatization to the news category dataset.

|                                                                                                                                                                       |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| for article in news\_df\['news\_summary']:     article = \[lemmatizer.lemmatize(token, "v") for token in article]     article = " ".join(article)     article.strip() |
