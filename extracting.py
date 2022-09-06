import requests
from bs4 import BeautifulSoup
import re

def extract_ing(url='https://world.openfoodfacts.org/entry-date/2016-08/ingredients'):

    """
    Extract ingredient names from the table in url
    """
   
    page = requests.get(url)
    soup = BeautifulSoup(page.text, "html.parser")
    r=re.compile('^[a-zA-Z\s-]*$') #
    ing=[]
    for hit in soup.findAll(attrs={'class' : 'tag known'}):
        if r.match(hit.text):
            #print("kalalal")
            ing.append(hit.text.lower())
            
    return ing

if __name__=="__main__":
   ing= extract_ing()
   print(ing)