# Netflix Titles Data Analysis Project

## Description
This project analyzes Netflix Movies and TV Shows dataset to derive insights and build recommendation systems. The analysis covers data cleaning, exploratory data analysis, and the implementation of three different recommender systems: CountVectorizer-based, TF-IDF with SVD, and a Weighted Autoencoder.

## Dataset
The dataset used in this project is obtained from [Kaggle](https://www.kaggle.com/datasets/rahulvyasm/netflix-movies-and-tv-shows).

## Project Structure
The project is organized into the following folders:
- `data`: Contains the dataset files.
  - [`netflix_titles.csv`](data/netflix_titles.csv): Original dataset file.
  - [`cleaned_netflix_titles.csv`](data/cleaned_netflix_titles.csv): Cleaned dataset file.
- `notebooks`: Contains Jupyter notebooks for data cleaning, analysis, and recommendation system implementation.
  - [`Data_Cleaning.ipynb`](notebooks/Data_Cleaning.ipynb): Notebook for data cleaning.
  - [`Dataset_Analysis.ipynb`](notebooks/Dataset_Analysis.ipynb): Notebook for exploratory data analysis.
  - [`Recommendation_System.ipynb`](notebooks/Recommendation_System.ipynb): Notebook for building recommendation systems.
- `src`: Contains Python scripts for the recommendation systems and the saved autoencoder model.
  - [`autoencoder_recommender.py`](src/autoencoder_recommender.py): Script for Autoencoder based recommender.
  - [`count_vectorizer_recommender.py`](src/count_vectorizer_recommender.py): Script for CountVectorizer based recommender.
  - [`tfidf_svd_recommender.py`](src/tfidf_svd_recommender.py): Script for TF-IDF with SVD based recommender.
  - [`autoencoder_recommender_model.h5`](src/autoencoder_recommender_model.h5): Saved Autoencoder model.

## Installation
To run this project, you need to have Python installed. Clone the repository and install the required packages:

```bash
git clone https://github.com/yourusername/netflix-titles-dataset-analysis.git
cd netflix-titles-dataset-analysis
pip install -r requirements.txt
```

## Usage

### Jupyter Notebooks
You can explore the Jupyter notebooks in the `notebooks` folder for data cleaning, exploratory data analysis, and building recommendation systems:

- [Data_Cleaning.ipynb](notebooks/Data_Cleaning.ipynb)
- [Dataset_Analysis.ipynb](notebooks/Dataset_Analysis.ipynb)
- [Recommendation_System.ipynb](notebooks/Recommendation_System.ipynb)

### Running Recommendation Systems
To run the recommendation systems using the Python scripts in the `src` folder, ensure you have the cleaned dataset in the `data` folder and the saved autoencoder model in the `src` folder.

For example, to use the CountVectorizer based recommender:

1. Open `src/count_vectorizer_recommender.py`.
2. Ensure the file path for the cleaned dataset (`data/cleaned_netflix_titles.csv`) is correctly set within the script.
3. Modify the script to specify the title for which you want recommendations.
4. Run the script.

```bash
python src/count_vectorizer_recommender.py
```

Repeat similar steps for the other recommenders:

- For TF-IDF with SVD based recommender: `src/tfidf_svd_recommender.py`
- For Autoencoder based recommender: `src/autoencoder_recommender.py`

## Results
The project demonstrates the following results:

- Data cleaning and preprocessing steps.
- Exploratory Data Analysis highlighting key insights from the Netflix Titles dataset.
- Implementation of three recommendation systems with their respective evaluations.

## Contributing
If you want to contribute to this project, please fork the repository and create a pull request with your changes.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

© 2024 Srinivasa Sai Damarla
