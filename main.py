from scraper.config_loader import ConfigLoader
from scraper.aaer_scraper import AAERScraper

# Load configuration from file
config = ConfigLoader.load_config('config.ini')

# Create an instance of the TableScraper class with page number 0
scraper = AAERScraper(0, config)

# Scrape the table and get the DataFrame
df = scraper.scrape_and_get_dataframe()

# Print the DataFrame
print(df)
