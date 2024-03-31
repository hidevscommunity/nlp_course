**Chapter 3 - NLP basic terminologies and their extraction**


Let’s discuss some basic terminologies commonly used in Natural Language Processing.

****

**3.1 Token/Word**

****

Words ****are the basic building blocks of a natural language and help in understanding the 

meaning/context of a sentence. In the NLP standard, words are referred to as tokens. Generally, the tokens are formed based on space.

****

For e.g. consider the sentence “This is an NLP crash course”. Here the tokens are “This”, “is”, “a”, “NLP”, “crash”, and “course”.

****

Now the question is how to extract these tokens programmatically.

****

**3.2 Tokenization**

****

**Why do we need Tokenization?**

****

Generally, we grasp the intention behind a text document(having lots of sentences) by reading it    sentence by sentence and processing each sentence word by word. In this way, we don’t lose the   essential information from the document. A word acts as a logical unit while understanding the document's intention. 

This similar approach we follow while processing/analyzing a document using NLP and this approach is known as **Tokenization**. 

****

Tokenization is the process of splitting the raw text into smaller chunks called tokens. 

Tokenization plays a significant role in understanding the meaning of a raw text.

****

So using tokenization we can easily split the text into words, this approach is called Word tokenization. Similarly, tokenization can be used to split a paragraph into sentences, then it is called sentence tokenization.

****

In Python, multiple tokenizers are available in different NLP libraries. The most commonly used 

tokenizers in NLTK are

a. sent\_tokenize

b. word\_tokenize

****

**a. sent\_tokenize**

****

sent\_tokenize() function is used to split a paragraph into sentences. It is available inside the nltk.tokenize module.

****

|                                                                                                                                                                                 |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| from nltk.tokenize import sent\_tokenize text = "This is a NLP crash course. Specially designed for beginners. It contains commonly used NLP techniques." sent\_tokenize(text)  |

****

![](https://lh7-us.googleusercontent.com/ygk8LQlKp1aR9_OjZ2Q69jFXq8MeVk8dMGikSqLFpSpkCqfu5cNFVKsBoiqs0PMQD5wkkAogisj9Ld13n1wU30CdkGG_bAPFhaayGro28QzUhMWoR6bBAf4m0miSXMwIaNP1dONPTkrjE_mCgbnJTTU)

As we can observe here, the paragraph is split into a sequence of 3 sentences.

**Tip -** The complete source code of this course can be downloaded from [here](https://drive.google.com/file/d/1zK3SEq0Hz24ZAKylwjhshoH-Hb-JxQzk/view?usp=share_link).

****

**b. word\_tokenize**

****

It splits the raw text into a sequence of words. It is available inside the same nltk.tokenize 

module.

****

|                                                                                                                                                                                 |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| from nltk.tokenize import word\_tokenize text = "This is a NLP crash course. Specially designed for beginners. It contains commonly used NLP techniques." word\_tokenize(text)  |

****

****![](https://lh7-us.googleusercontent.com/QBVvICgDKO0RUw9WFWP2rZ24m7T1AlkWLvfDx19OkVE7iR2jCPWxRv1qxe4GAAqscXcnoqpeSkCQHduE2G5Q9XI1w9924g5z1Ft5SCBZpdhPexwijMbnOwH61zTqVciRy7MzvIlRur1bqX2WJWHjcSM)****

****

Here, the input raw text is split into a sequence of words.

****

**3.3 Part of Speech(PoS) tagging**

Part-of-Speech (PoS) tagging is the process of assigning labels(parts of speech) to 

individual words in a sentence, based on their context and definition. It is helpful in 

extracting relationships among words and building a knowledge graph. 

****

A word can have different meanings in different sentences depending 

on the context. For e.g. the word “plant” appears as a noun in “Japan nuclear firm 

shuts plants” whereas it appears as a verb in “I planted trees”.

****

Common parts of speech include nouns, pronouns, verbs, adverbs, adjectives, prepositions, 

conjunction, interjections etc. 

Using NLTK’s built-in pos\_tag() method we can easily extract Part-of-Speech from a sentence. 

****

Consider the news title “Japan nuclear firm shuts plants” from the business news 

category. Let’s see how to extract Part-of-Speech from it.

****

|                                                                                                          |
| -------------------------------------------------------------------------------------------------------- |
| import nltk text = "Japan nuclear firm shuts plants" tokens = word\_tokenize(text) nltk.pos\_tag(tokens) |

![](https://lh7-us.googleusercontent.com/ulWW_IU9XtyWSKOHaDpnoU-euaZKQjf5EgX6OMbqQyS5k-f2sFPYelNavQpXahaC2Y2JS48JbMr_8xazJ9Y_7zSr7d1HkrWG2o-RWvcnYIO0F5O-MEjkSUErj1rQ_l0jadjMu5QQA4ktAb86B6Prx1A)

Here, we can observe that

“Japan” is tagged as NNP(proper noun, singular).

“nuclear” is tagged as JJ(adjective).

 “firm” is tagged as NN(noun, singular).

“shuts” is tagged as VBZ(Verb, 3rd person singular).

“plants” is tagged as NNS(noun, plural). 

Note - The complete list of pos tags and their description can be found [here](https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html).

Similarly, for the sentence  “I planted trees” the pos can be extracted as follows.

****

|                                                                              |
| ---------------------------------------------------------------------------- |
| text = "I planted trees" tokens = word\_tokenize(text) nltk.pos\_tag(tokens) |

![](https://lh7-us.googleusercontent.com/Tm2bovcLIdKB38aMRi1gtMUD2dFy6ANsZuuOnv8qdtL-Zkgak4S_1iV5gptkziwXtzZ6TLcTkqU9SM5SOaRjWwhVVEA8eBhYIMgbeStU89oKw4wtazqI8Odj6z9S0bWoZTuzQQtQHCgMI47dIlra758)

**3.4 Named Entity Recognition(NER)**


Identifying names of entities(like a person, organization, location) from the raw text can help in multiple conclusions. Recognizing such predefined named entities (like names of persons, organizations, locations, monetary values, quantities etc.) can guide in extracting meaningful information from many real-world raw texts. 

For e.g., using NER we can answer the following questions from the news category dataset.

- Which organizations/companies are mentioned in the “Business News” articles?

- Who all athletes are mentioned in the “Sports News” articles?

- What products/services are mentioned in “Science-Technology News” articles?

Therefore, NER is very helpful in information extraction. Now let’s see how to extract NER using NLTK.  

****

NLTK has a pre-trained named entity chunker i.e. ne\_chunk() method. This method is available in the nltk.chunk module, it accepts the POS-tagged sentence as input and chunks the sentence into a tree.

_Tip - \*\*Chunks are made up of tokens and chunking is the process of grouping the tokens into chunks based on some pattern._

****

|                                                                                                                                                                                                        |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| from nltk.chunk import ne\_chunk text = "Biden is America's president" tokens = word\_tokenize(text) tagged = nltk.pos\_tag(tokens) res\_chunk = nltk.ne\_chunk(tagged, binary=True) res\_chunk.draw() |

![](https://lh7-us.googleusercontent.com/t8Aiht7N7hT1qTWUE0wpCUYZ9fkqV025j57zmzYe3NATd21vMujq76iYp4rV06Ao2MagQrQ8Hlip11WhQutJtw7PhsceLB-fogw3mw1OTOi3uugO7P8abUq_W7HBKlbBSVONiOiouxw1bK1eJO_9J7g)

As we can observe here, the sentence S (“Trump is America’s president”) is divided into chunks and represented as a tree. The NE (person name and location name) are “Trump” and “America” respectively.
