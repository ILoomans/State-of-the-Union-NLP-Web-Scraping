import requests
import urllib.request
import time
from bs4 import BeautifulSoup
import pandas as pd

# Set the URL you want to webscrape from
url = 'https://www.presidency.ucsb.edu/documents/app-categories/spoken-addresses-and-remarks/presidential/state-the-union-addresses?items_per_page=100'

# Connect to the URL
response = requests.get(url)


# Parse HTML and save to BeautifulSoup object¶
soup = BeautifulSoup(response.text, "html.parser")
x = soup.find('div',class_='view-content')
x.find_all('a')



democratic =['Barack Obama','William J. Clinton','Jimmy Carter','Lyndon B. Johnson', 'John F. Kennedy']
republican = ['Donald J. Trump','George W. Bush','George Bush', 'Ronald Reagan','Gerald R. Ford','Richard Nixon', 'Dwight D. Eisenhower']

#Republicans are 1 
#Democrats are 0

base = 'https://www.presidency.ucsb.edu'
data = pd.DataFrame(columns=['President','Speech'])
datalist = []
for t in x.find_all('a'):
    link = t['href']
    if('/documents/' in link):
        download_url = base+link
        subresponse = requests.get(download_url)
        subsoup = BeautifulSoup(subresponse.text, "html.parser")
        ss = subsoup.find('h3',class_='diet-title')
        sa = ss.get_text()
#        print(sa.contents[0])
        st = subsoup.find('div',class_='field-docs-content')
#        print(st.get_text())
        pp = 0 if sa in democratic else 1
        datalist.append([sa,st.get_text(),pp])
        print(download_url)
        
data = pd.DataFrame(datalist,columns=['President','Speech','Party'])
data = data[:60]


data.to_csv('sotu.csv')
total = 0   
for x in data['Speech']:
    total = total + len(x)
x.find('a')
sub  = BeautifulSoup(x, "html.parser")