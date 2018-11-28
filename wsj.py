# Author: Eliska Kloberdanz
# import libraries

from bs4 import BeautifulSoup
import urllib.request
from urllib.parse import urljoin
import requests
import re
from datetime import date, timedelta
import pandas as pd
import pprint


startDate = date(2018, 10, 24)
endDate = date (2018, 10, 26)

numberOfDays = endDate - startDate
listOfDays = []
for i in range(numberOfDays.days + 1):
    listOfDays.append(startDate + timedelta(i))

weekdays = [0, 1, 2, 3, 4]
listOfWeekDays = []
for date in listOfDays:
    if date.weekday() in weekdays:
        listOfWeekDays.append(date)


site = 'http://www.wsj.com/public/page/archive-'


ListOfArticles = []
    
for day in listOfWeekDays:
    day = str(day)
    r = requests.get(site+day+'.html')
    headlines = r.text
    souplink = BeautifulSoup(headlines, 'lxml')
    tags = souplink.findAll('p')
    DayText = []
    for t in tags:
        DayText.append(t.text)
    ListOfArticles.append(DayText)
     
        
        #print(t.text)
#articleWords = []
#sampleDay = ListOfArticles[1]
#print(sampleDay)
#for sentence in sampleDay:
    #articleWords.append(sentence.split())

#print(articleWords)

#print(len(ListOfArticles))
#print(len(listOfWeekDays))




import nltk
import os
import re
import pprint
#nltk.download('punkt')
from nltk.tokenize import word_tokenize

# Step 1 – Training data

base_directory = '/home/eliska/Iowa State/Principles of AI/Stock-Picker/MovieTrain/debug'
sentences = []
for directory in ['pos', 'neg']:
    full_directory = os.path.join(base_directory, directory)
    for filename in os.listdir(full_directory):
        filename = os.path.join(full_directory, filename)
        with open(filename, 'r') as f:
            paragraph = f.read()
            for sentence in ((sentence.strip(), directory)
                             for sentence in re.split('\.|!|\?', paragraph) if sentence):
                sentences.append(sentence)

train = sentences

# Step 2 
dictionary = set(word.lower() for passage in train for word in word_tokenize(passage[0]))
  
# Step 3
t = [({word: (word in word_tokenize(x[0])) for word in dictionary}, x[1]) for x in train]
  
# Step 4 – the classifier is trained with sample data
classifier = nltk.NaiveBayesClassifier.train(t)

listOfRatings = []
for article in ListOfArticles:
    sumOfPositive = 0
    sumOfNegative = 0
    for sentence in article:
        test_data = sentence.split('.')
        for test in test_data:
            test_data_features = {word.lower(): (word in word_tokenize(test.lower())) for word in dictionary}
  
            result = classifier.classify(test_data_features)
            if result == "pos":
                sumOfPositive += 1
            else:
                sumOfNegative += 1
    if sumOfPositive > sumOfNegative:
        listOfRatings.append("pos")
    else:
        listOfRatings.append("neg")

data = {'Date':listOfWeekDays, 'Article': ListOfArticles, 'Sentiment': listOfRatings}

dataFrame = pd.DataFrame(data)

dataFrame.to_csv('wsjTest.csv')
