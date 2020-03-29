# Spam-Detection-NLP
Spam Detection using a simple Natural Language Machine Learning Pipeline

We'll be using a dataset from the UCI datasets! 

The file we are using contains a collection of more than 5 thousand SMS phone messages

A collection of texts is also sometimes called "corpus".  Printing the first ten messages gives us:

0 ham	Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...

1 ham	Ok lar... Joking wif u oni...

2 spam	Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's

3 ham	U dun say so early hor... U c already then say...

4 ham	Nah I don't think he goes to usf, he lives around here though

5 spam	FreeMsg Hey there darling it's been 3 week's now and no word back! I'd like some fun you up for it still? Tb ok! XxX std chgs to send, £1.50 to rcv

6 ham	Even my brother is not like to speak with me. They treat me like aids patent.

7 ham	As per your request 'Melle Melle (Oru Minnaminunginte Nurungu Vettam)' has been set as your callertune for all Callers. Press *9 to copy your friends Callertune


8 spam	WINNER!! As a valued network customer you have been selected to receivea £900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only.

9 spam	Had your mobile 11 months or more? U R entitled to Update to the latest colour mobiles with camera for Free! Call The Mobile Update Co FREE on 08002986030

## Using these labeled ham and spam examples, we'll train a machine learning model to learn to discriminate between ham/spam automatically. Then, with a trained model, we'll be able to classify arbitrary unlabeled messages as ham or spam.

## Text Preprocessing
### Bag of Words with CountVectorizer
We convert each message of the corpus into a Bag of Words dataframe using CountVectorizer() module, where each column is a Message index, and each row is for a row, each row's cell value indicates the number of times it appears in that message, after we remove "STOP" words (commonly used English language words that don't hold any useful meaning, such as articles, prepositions, pronouns, ect.

### TDIDIF
After the counting, the term weighting and normalization can be done with TF-IDF, using scikit-learn's TfidfTransformer.
TF-IDF stands for term frequency-inverse document frequency, and the tf-idf weight is a weight often used in information retrieval and text mining. This weight is a statistical measure used to evaluate how important a word is to a document in a collection or corpus. The importance increases proportionally to the number of times a word appears in the document but is offset by the frequency of the word in the corpus. Variations of the tf-idf weighting scheme are often used by search engines as a central tool in scoring and ranking a document's relevance given a user query.

## Naive Bayes Classifier
With messages represented as vectors, we can finally train our spam/ham classifier. Now we can actually use almost any sort of classification algorithms. For a variety of reasons, the Naive Bayes classifier algorithm is a good choice.

`spam_detect_model = MultinomialNB().fit(messages_tfidf, messages['label'])`

`all_predictions = spam_detect_model.predict(messages_tfidf)
print(all_predictions)`

['ham' 'ham' 'spam' ..., 'ham' 'ham' 'ham'] 

## Results

             precision    recall  f1-score   support

        ham       0.98      1.00      0.99      4825
       spam       1.00      0.85      0.92       747

avg / total       0.98      0.98      0.98      5572

![](700px-Precisionrecall.svg)


