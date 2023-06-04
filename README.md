# AAER Scraper

AAER Scraper is a Python application that allows you to scrape table data from a webpage and store it in a Pandas DataFrame. It utilizes the requests library for making HTTP requests, BeautifulSoup for HTML parsing, and Pandas for data manipulation.

Certainly! Here's the updated folder structure section of the README.md file:

## Folder Structure

The folder structure of the AAER Scraper project is as follows:

```
- AAER_Data.csv
- AAER_Data_2014_Onwards.csv
- AAER_Scraper.ipynb
- CODE_OF_CONDUCT
- LICENSE
- README.md
- config.ini
- http_cache.sqlite
- main.py
- requirements.txt
- scraper/
  - __init__.py
  - aaer_scraper.py
  - config_loader.py
```

- `AAER_Data.csv` and `AAER_Data_2014_Onwards.csv`: Sample CSV files containing scraped data.
- `AAER_Scraper.ipynb`: Jupyter Notebook containing an example usage of the AAER Scraper.
- `CODE_OF_CONDUCT`: Community code of conduct guidelines.
- `LICENSE`: License information for the AAER Scraper.
- `README.md`: Readme file providing an overview of the AAER Scraper and instructions for installation and usage.
- `config.ini`: Configuration file containing necessary values for the AAER Scraper.
- `http_cache.sqlite`: SQLite database file used for caching HTTP requests.
- `main.py`: Python script for running the AAER Scraper.
- `requirements.txt`: File specifying the required dependencies for the AAER Scraper.
- `scraper/`: Directory containing the source code for the AAER Scraper.

Please note that this is just a sample folder structure and may vary based on your specific implementation or usage requirements.

This updated section provides a more detailed description of each file and directory in the folder structure and their purposes within the project.

## Installation

1. Clone the repository:

   ```shell
   git clone https://github.com/pChitral/Accounting-and-Auditing-Enforcement-Releases-Web-Scraper.git
   ```

2. Install the required dependencies using pip:

   ```shell
   pip install -r requirements.txt
   ```

## Usage

1. Update the `config.ini` file:

   The `config.ini` file contains the necessary configuration values for the AAER Scraper. Update the following values as per your requirements:

   - `base_url`: The base URL of the webpage to scrape.
   - Any other configuration values specific to your use case.

2. Run the `main.py` script:

   ```python
   python main.py
   ```

   The script will execute the scraping process and store the scraped data in a Pandas DataFrame.

   **Note**: Ensure that you have the necessary permissions to access the webpage and that it contains a table with the required structure.

## Configuration

The `config.ini` file contains the configuration values for the AAER Scraper. Update these values according to your needs. Below is an explanation of the available options:

- `base_url`: The base URL of the webpage to scrape. Make sure to include the necessary query parameters if required.

## Contributing

Contributions to AAER Scraper are welcome! If you find any issues or have suggestions for improvements, please feel free to open an issue or submit a pull request. When contributing, please follow the existing coding style and guidelines.

## License

AAER Scraper is released under the [MIT License](LICENSE).

This version of the README.md file incorporates improved formatting using code blocks and indented directory structure. It is more visually appealing when viewed on GitHub.
