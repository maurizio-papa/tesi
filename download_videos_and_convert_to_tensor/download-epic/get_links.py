import requests
from bs4 import BeautifulSoup

def scrape_links(url):
    '''
    Scrape links from the specified URL.
    Args:
        url (str): URL of the webpage to scrape.
    Returns:
        links (list): List of links found on the webpage.
    '''
    links = []

    try:
        # Send a GET request to the URL
        response = requests.get(url)
        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')
        # Find all anchor tags (links) in the webpage
        for link in soup.find_all('a'):
            href = link.get('href')
            if href and href.startswith('http'):  # Filter out only the HTTP/HTTPS links
                links.append(href)
    except Exception as e:
        print(f"An error occurred: {e}")

    return links

# Example usage:
url_to_scrape = 'https://data.bris.ac.uk/data/dataset/0d8372d37ec74e2f7c14746f582fddcd'
scraped_links = scrape_links(url_to_scrape)
print(scraped_links)