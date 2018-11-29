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
import pickle
import os


articlesDirectory = '/home/eliska/Iowa State/Principles of AI/Stock-Picker/Articles'
listOfRatings = []

f = open('naivebayes1.pickle', 'rb')
classifier = pickle.load(f)

for filename in os.listdir(articlesDirectory):
    filename = os.path.join(articlesDirectory, filename)
    sumOfPositive = 0
    sumOfNegative = 0
    
    with open(filename, 'r') as f:
        for line in f:
            feats = dict([(word, True) for word in line])
      
            result = classifier.classify(feats)
                
                
        if result == "pos":
            sumOfPositive += 1
        else:
            sumOfNegative += 1
    if sumOfPositive > sumOfNegative:
        listOfRatings.append("pos")
    else:
        listOfRatings.append("neg")


#print(listOfRatings)
f.close()

data = {'Date':listOfWeekDays, 'Article': ListOfArticles, 'Sentiment': listOfRatings}

dataFrame = pd.DataFrame(data)

dataFrame.to_csv('wsjSentiment.csv')



