
# import libraries
from bs4 import BeautifulSoup
import urllib.request
from urllib.parse import urljoin
import requests


url = "https://www.reuters.com/"
 
# Getting the webpage, creating a Response object.
response = requests.get(url)
 
# Extracting the source code of the page.
data = response.text
 
# Passing the source code to BeautifulSoup to create a BeautifulSoup object for it.
soup = BeautifulSoup(data, 'lxml')
 
 # Extract all links (absolute paths)
links = []
for link in soup.findAll('a'):
    links.append(urljoin(url,link.get('href')))
    #links.append(link.get('href'))

#print(links)

def read_page(page):
    print('openening: {}'.format(page))
    try:
        response = urllib.request.urlopen(page)
        s = response.read()
        return s.decode()
    except urllib.error.HTTPError as e:
        return ''

#print(read_page(url))


# Keywords of interest
keyWords = ["interest rate", "employment", "inflation", "GDP", "earnings", "S&P", "inflation", "Fed", "federal reserve", "oil", "market", "stocks", "surge", "decline", "trade"]

# Links of interest based on keywords
linkswithKeyWords = []
for link in links:
    for word in keyWords:
        if word in link:
            linkswithKeyWords.append(link)
        
#print(linkswithKeyWords)

print(read_page(linkswithKeyWords[1]))
'''

for link in linkswithKeyWords:
    read_page(link)
    
    


