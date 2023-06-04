import requests
from bs4 import BeautifulSoup
import pandas as pd
from urllib.parse import urljoin
import logging
import requests_cache
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


class AAERScraper:
    def __init__(self, page, config):
        """
        Initializes the AAERScraper class.

        Args:
            page (int): The page number to scrape.
            config (ConfigParser): The configuration parser object.
        """
        self.base_url = config.get('Scraper', 'base_url')
        self.page = page
        self.url = self.construct_url()
        self.data = []

    def construct_url(self):
        """
        Constructs the URL for scraping based on the page number.

        Returns:
            str: The constructed URL.
        """
        params = {"page": self.page}
        url = urljoin(self.base_url, "?".join(
            [self.base_url, "&".join(f"{k}={v}" for k, v in params.items())]))
        return url

    def fetch_html_content(self):
        """
        Fetches the HTML content of the webpage.
        """
        try:
            response = self.session.get(self.url)
            response.raise_for_status()  # Raise an exception if the request was not successful
            self.html_content = response.content
        except requests.RequestException as e:
            logging.error(f"Error occurred while fetching HTML content: {e}")

    def parse_html_content(self):
        """
        Parses the HTML content using BeautifulSoup.
        """
        soup = BeautifulSoup(self.html_content, 'html.parser')
        self.table = soup.find('table', class_='list')

    def scrape_table_data(self):
        """
        Scrapes the table data and stores it in a list of dictionaries.
        """
        if self.table is None:
            logging.warning("No table found on the page.")
            return

        rows = self.table.select('.pr-list-page-row')

        for row in rows:
            self.data.append(self.extract_row_data(row))

    def extract_row_data(self, row):
        """
        Extracts data from a table row and returns a dictionary.

        Args:
            row (BeautifulSoup): The BeautifulSoup object representing a table row.

        Returns:
            dict: The extracted data as a dictionary.
        """
        date_element = row.select_one('.datetime')
        date = date_element.text.strip() if date_element else None

        respondents_element = row.select_one('.release-view__respondents')
        respondents = respondents_element.text.strip() if respondents_element else None

        release_numbers_element = row.select_one(
            '.view-table_subfield.view-table_subfield_release_number')
        release_numbers = release_numbers_element.text.replace(
            'Release No.', '').strip() if release_numbers_element else None

        link_element = respondents_element.find('a')
        link = urljoin(
            self.base_url, link_element['href']) if link_element else None

        return {'Date': date, 'Respondents': respondents, 'Release Numbers': release_numbers, 'Link': link}

    def create_dataframe(self):
        """
        Creates a DataFrame from the scraped data.

        Returns:
            pd.DataFrame: The DataFrame containing the scraped data.
        """
        df = pd.DataFrame(self.data)
        return df

    def save_to_csv(self, filename):
        """
        Saves the scraped data to a CSV file.

        Args:
            filename (str): The name of the CSV file to save.
        """
        df = self.scrape_and_get_dataframe()
        df.to_csv(filename, index=False)
        logging.info(f"Data saved to {filename} successfully.")

    def scrape_and_get_dataframe(self):
        """
        Performs the entire scraping process and returns the DataFrame.

        Returns:
            pd.DataFrame: The DataFrame containing the scraped data.
        """
        self.session = requests_cache.CachedSession(
            backend='memory', expire_after=3600)
        self.fetch_html_content()
        self.parse_html_content()
        self.scrape_table_data()
        df = self.create_dataframe()
        return df

    def handle_pagination(self, num_pages=1, max_workers=5):
        """
        Handles scraping multiple pages of data and concatenates the results using concurrent requests.

        Args:
            num_pages (int, optional): The number of pages to scrape. Defaults to 1.
            max_workers (int, optional): The maximum number of worker threads to use for concurrent requests.
                                         Defaults to 5.

        Returns:
            pd.DataFrame: The DataFrame containing the concatenated data.
        """
        all_data = []  # List to store data from all pages

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_page = {executor.submit(self.scrape_page, page): page for page in range(
                self.page, self.page + num_pages)}
            for future in concurrent.futures.as_completed(future_to_page):
                page = future_to_page[future]
                try:
                    page_data = future.result()
                    all_data.extend(page_data)
                    logging.info(f"Scraped page {page} successfully.")
                except Exception as e:
                    logging.error(
                        f"Error occurred while scraping page {page}: {e}")

        df = pd.DataFrame(all_data)
        return df

    def scrape_page(self, page):
        """
        Scrapes a single page and returns the data as a list of dictionaries.

        Args:
            page (int): The page number to scrape.

        Returns:
            list: The scraped data as a list of dictionaries.
        """
        self.page = page
        self.url = self.construct_url()
        self.fetch_html_content()
        self.parse_html_content()
        self.data = []  # Clear the data list for each page
        self.scrape_table_data()
        return self.data
