from icrawler.builtin import GoogleImageCrawler
from bing_image_downloader import downloader

# Define download keywords and output directory
keywords = ["yoga", "running", "cycling"]
output_root = "data/raw/"

# Google Image Scraper
def scrape_google_images():
    for keyword in keywords:
        google_crawler = GoogleImageCrawler(storage={"root_dir": f"{output_root}/{keyword}"})
        google_crawler.crawl(keyword=keyword, max_num=500)

# Bing Image Scraper
def scrape_bing_images():
    for keyword in keywords:
        downloader.download(keyword, limit=500, output_dir=output_root, adult_filter_off=True, force_replace=False, timeout=60)

if __name__ == "__main__":
    print("Starting Google image scraping...")
    scrape_google_images()
    print("Google image scraping complete.")

    print("Starting Bing image scraping...")
    scrape_bing_images()
    print("Bing image scraping complete.")

    print("All images downloaded successfully!")