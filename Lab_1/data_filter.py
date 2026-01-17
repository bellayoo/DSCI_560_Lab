from bs4 import BeautifulSoup
import requests
import csv

with open("../data/raw_data/web_data.html", "r") as file:
    soup = BeautifulSoup(file, "html.parser")


market_data = []
news_data = []

print("Filtering Market Fields")
market_container = soup.select_one("#market-data-scroll-container")
market_cards = market_container.find_all("a", class_="MarketCard-container")

print("Stroing Market Data")
for i in market_cards:
	symbol = i.select_one(".MarketCard-symbol")
	position = i.select_one(".MarketCard-stockPosition")
	change_pct = i.select_one(".MarketCard-changePct")
	market_data.append([symbol.get_text(), position.get_text(), change_pct.get_text()])

print("Saving Market Data as CSV")
with open("../data/processed_data/market_data.csv", "w", newline="") as file:
	writer = csv.writer(file)
	writer.writerow(["symbol","position","change_pct"])
	writer.writerows(market_data)


print("Filtering News Fields")
news_container = soup.select_one("ul.LatestNews-list")
news_items = news_container.find_all("li")

print("Stroing News Data")
for i in news_items:
	timestamp = i.select_one("time.LatestNews-timestamp").get_text()
	title = i.select_one("a.LatestNews-headline").get_text()
	link = i.select_one("a.LatestNews-headline")["href"]
	news_data.append([timestamp, title, link])

print("Saving News Data as CSV")
with open("../data/processed_data/news_data.csv", "w", newline="") as file:
	writer = csv.writer(file)
	writer.writerow(["timestamp","title","link"])
	writer.writerows(news_data)

print("Done")
