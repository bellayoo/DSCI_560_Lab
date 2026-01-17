from bs4 import BeautifulSoup
import requests

url = "http://www.cnbc.com/world/?region=world" 

headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"}
response = requests.get(url, headers=headers)

#headers reference: https://www.zenrows.com/blog/user-agent-web-scraping#how-to

# tags for "Market" banner: span
#<div id="market-data-scroll-container" class="MarketsBanner-marketData">

# tags for Latest News: div
#<div class="LatestNews-isHomePage LatestNews-isIntlHomepage" data-test="latestNews-0" data-analytics="HomePageInternational-latestNews-7-0">

soup = BeautifulSoup(response.text, "html.parser").prettify()
outpath = "../data/raw_data/web_data.html"
with open(outpath, "w") as f:
  f.write(soup)
