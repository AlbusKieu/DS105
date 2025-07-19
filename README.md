# Movie Recommendation System

This project builds a movie recommendation system using data from IMDB and Rotten Tomatoes. The system includes data crawling, preprocessing, data analysis, building recommendation models, and a web interface with Streamlit.

## Main File Functions

- **IMDB_crawler.ipynb**: Crawls movie data from IMDB and saves it to a CSV file.
- **ROTTEN_TOMATO_crawler.ipynb**: Crawls movie data from Rotten Tomatoes and saves it to a CSV file.
- **main.ipynb**:
  - Reads data from the crawled files.
  - Data preprocessing: normalization, encoding (LabelEncoder), genre processing, missing value handling.
  - Exploratory Data Analysis (EDA): statistics, data visualization.
  - Builds the movie recommendation system with the following models/algorithms:
    - **User-based Filtering**: Uses Collaborative Filtering based on the user-item matrix, calculates user similarity using Cosine Similarity or Pearson Correlation, and recommends movies that similar users have highly rated.
    - **Content-based Filtering**: Uses movie content features (genre, description, actors, directors, etc.), vectorizes movie features using TF-IDF or CountVectorizer, calculates similarity between movies using Cosine Similarity, and recommends movies similar to those the user liked.
    - **Hybrid Approach**: Combines both methods, can use the average or weighted score between User-based and Content-based recommendations, or use machine learning models (e.g., Linear Regression) to combine features from both methods for optimal recommendations.
  - Evaluates models using metrics such as Precision, Recall, RMSE, etc.
  - Outputs recommendations for each user or specific queries.
- **app.py**: Builds a web interface with Streamlit, allowing users to input information and receive movie recommendations directly.

## Project Structure

```
├── app.py                        # Streamlit web application
├── main.ipynb                    # Main notebook: preprocessing, EDA, building recommendation models
├── IMDB_crawler.ipynb            # Notebook for crawling IMDB data
├── ROTTEN_TOMATO_crawler.ipynb   # Notebook for crawling Rotten Tomatoes data
├── demo_info_from_link.ipynb     # Demo notebook for extracting data from existing links
├── Preprocessing_EDA.ipynb       # Demo notebook for preprocessing and data modeling methods
```

## Usage Guide

### 1. Clone the repository

```bash
git clone https://github.com/AlbusKieu/DS105.git
cd DS105
```

### 2. Install required libraries

It is recommended to use a virtual environment:

```bash
pip install pandas numpy scikit-learn streamlit requests beautifulsoup4
```

### 3. Crawl data

- Open and run the following notebooks in order:
  - `IMDB_crawler.ipynb`: Crawl movie data from IMDB.
  - `ROTTEN_TOMATO_crawler.ipynb`: Crawl movie data from Rotten Tomatoes.

The results will be saved to the corresponding CSV files.

### 4. Preprocess, analyze, and build recommendation models

- Open and run `main.ipynb`:
  - Read data from the collected CSV files.
  - Preprocess data: normalization, encoding, genre processing, missing value handling.
  - Perform EDA: statistics, data visualization.
  - Build movie recommendation models using machine learning algorithms:
    - **User-based Filtering**: Collaborative Filtering (Cosine Similarity, Pearson Correlation).
    - **Content-based Filtering**: TF-IDF/CountVectorizer, Cosine Similarity.
    - **Hybrid Approach**: Combine both methods, possibly with Linear Regression or weighted average.
  - Evaluate models and output recommendations.

### 5. Run the web interface with Streamlit

```bash
streamlit run app.py
```

Then open the displayed URL in your terminal to use the movie recommendation system online.

## Notes

- Make sure to finish the data crawling steps before running `main.ipynb` or `app.py`.

---

## License
This project is for educational purposes.
