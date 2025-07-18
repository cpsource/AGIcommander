from bs4 import BeautifulSoup
import requests

html = requests.get("https://example.com").text
soup = BeautifulSoup(html, 'html.parser')  # or 'lxml', 'html5lib'

