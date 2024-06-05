import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.csr import csr_matrix
from typing import List, Tuple

class ContentBasedRecommendationSystem:
    """
    A recommendation system for Netflix titles based on content similarity using Count Vectorizer and cosine similarity.
    """

    def __init__(self, file_path: str) -> None:
        """
        Initialize the ContentBasedRecommendationSystem with the dataset file path.

        Params:
        - file_path (str): Path to the cleaned Netflix titles CSV file.
        """
        self.df: pd.DataFrame = pd.read_csv(file_path)
        self.df['combined_features']: pd.Series = self._combine_features()
        self.count_matrix: csr_matrix = self._create_count_matrix()
        self.cosine_sim: csr_matrix = self._compute_cosine_similarity()

    def _combine_features(self) -> pd.Series:
        """
        Combine relevant features into a single string for each title.

        Returns:
        - pd.Series: A series with combined feature strings.
        """
        return (
            self.df['title'] + ' ' + self.df['director'].fillna('') + ' ' +
            self.df['cast'].fillna('') + ' ' + self.df['country'].fillna('') + ' ' +
            self.df['listed_in'] + ' ' + self.df['description'] + ' ' +
            self.df['release_year'].astype(str) + ' ' + self.df['rating'].fillna('')
        )

    def _create_count_matrix(self) -> csr_matrix:
        """
        Create a count matrix from the combined features.

        Returns:
        - csr_matrix: A matrix of token counts.
        """
        count_vectorizer: CountVectorizer = CountVectorizer(stop_words='english')
        return count_vectorizer.fit_transform(self.df['combined_features'])

    def _compute_cosine_similarity(self) -> csr_matrix:
        """
        Compute the cosine similarity matrix from the count matrix.

        Returns:
        - csr_matrix: A matrix of cosine similarity scores.
        """
        return cosine_similarity(self.count_matrix, self.count_matrix)

    def _get_similar_indices(self, title_idx: int, num_recommendations: int) -> List[Tuple[int, float]]:
        """
        Get indices of titles similar to the given title index.

        Params:
        - title_idx (int): The index of the title to base recommendations on.
        - num_recommendations (int): The number of recommendations to return.

        Returns:
        - List[Tuple[int, float]]: List of tuples containing index and similarity score.
        """
        sim_scores: List[Tuple[int, float]] = list(enumerate(self.cosine_sim[title_idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        return sim_scores[1:num_recommendations + 1]

    def get_recommendations(self, title: str, num_recommendations: int = 10) -> pd.DataFrame:
        """
        Get movie recommendations based on the similarity to a given title.

        Params:
        - title (str): The title of the movie to base recommendations on.
        - num_recommendations (int): The number of recommendations to return (default is 10).

        Returns:
        - pd.DataFrame: DataFrame of recommended titles.
        """
        idx: int = self.df[self.df['title'] == title].index[0]
        similar_indices: List[Tuple[int, float]] = self._get_similar_indices(idx, num_recommendations)
        title_indices: List[int] = [i[0] for i in similar_indices]
        similarity_scores: List[float] = [i[1] for i in similar_indices]
        recommendations: pd.DataFrame = self.df.iloc[title_indices].copy()
        recommendations['similarity_score'] = similarity_scores
        return recommendations

    def get_movie_details(self, title: str) -> pd.DataFrame:
        """
        Get the details of a given movie.

        Params:
        - title (str): The title of the movie to display.

        Returns:
        - pd.DataFrame: DataFrame of the given movie's details.
        """
        return self.df[self.df['title'] == title]