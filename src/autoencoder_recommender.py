import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import History
from typing import Tuple, List

class WeightedAutoencoderRecommendationSystem:
    """
    A recommendation system for Netflix titles based on content similarity using a simple Autoencoder with weighted features.
    """

    def __init__(self, file_path: str, encoding_dim: int = 200, handle_placeholders: str = 'ignore') -> None:
        """
        Initialize the WeightedAutoencoderRecommendationSystem with the dataset file path.

        Params:
        - file_path (str): Path to the cleaned Netflix titles CSV file.
        - encoding_dim (int): Dimension of the encoding layer in the Autoencoder.
        - handle_placeholders (str): Strategy to handle placeholders ('ignore' or 'keep').
        """
        self.file_path = file_path
        self.encoding_dim = encoding_dim
        self.handle_placeholders = handle_placeholders
        self.df: pd.DataFrame = pd.read_csv(file_path)
        self.df['combined_features']: pd.Series = self._combine_features()
        self.tfidf_matrix: np.ndarray = self._create_tfidf_matrix()
        self.svd_matrix: np.ndarray = self._apply_svd(self.tfidf_matrix)
        self.encoded_matrix: np.ndarray = np.array([])
        self.cosine_sim: np.ndarray = np.array([])
        self.autoencoder: Model = self._build_autoencoder(self.svd_matrix.shape[1], encoding_dim)

    def _combine_features(self) -> pd.Series:
        """
        Combine relevant features into a single string for each title with weights.

        Returns:
        - pd.Series: A series with combined feature strings.
        """
        def repeat_text(text: str, weight: float) -> str:
            """Repeat the text based on the weight"""
            count = max(1, int(weight * 10))
            return (text + ' ') * count

        def process_field(field: pd.Series, weight: float) -> pd.Series:
            if self.handle_placeholders == 'ignore':
                return field.apply(lambda x: repeat_text(x.lower(), weight) if x.lower() != 'unknown' else '')
            return field.apply(lambda x: repeat_text(x.lower(), weight))

        combined_features = (
            process_field(self.df['title'], 0.5) + ' ' +
            process_field(self.df['director'].fillna('unknown'), 0.3) + ' ' +
            process_field(self.df['cast'].fillna('unknown'), 0.3) + ' ' +
            process_field(self.df['country'].fillna('unknown'), 0.1) + ' ' +
            process_field(self.df['listed_in'], 0.4) + ' ' +
            process_field(self.df['description'], 0.3) + ' ' +
            process_field(self.df['release_year'].astype(str), 0.2) + ' ' +
            process_field(self.df['rating'].fillna('unknown'), 0.2)
        )

        return combined_features

    def _create_tfidf_matrix(self) -> np.ndarray:
        """
        Create a TF-IDF matrix from the combined features.

        Returns:
        - np.ndarray: A matrix of TF-IDF features.
        """
        tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf_vectorizer.fit_transform(self.df['combined_features'])
        return tfidf_matrix

    def _apply_svd(self, tfidf_matrix: np.ndarray) -> np.ndarray:
        """
        Apply TruncatedSVD to reduce the dimensionality of the TF-IDF matrix.

        Params:
        - tfidf_matrix (np.ndarray): The TF-IDF feature matrix.

        Returns:
        - np.ndarray: The reduced feature matrix.
        """
        svd = TruncatedSVD(n_components=200)
        svd_matrix = svd.fit_transform(tfidf_matrix)
        return svd_matrix

    def _build_autoencoder(self, input_dim: int, encoding_dim: int) -> Model:
        """
        Build a simple Autoencoder model.

        Params:
        - input_dim (int): Dimension of the input layer.
        - encoding_dim (int): Dimension of the encoding layer.

        Returns:
        - Model: A compiled Autoencoder model.
        """
        input_layer = Input(shape=(input_dim,))
        encoded = Dense(encoding_dim, activation='relu')(input_layer)
        decoded = Dense(input_dim, activation='sigmoid')(encoded)
        autoencoder = Model(input_layer, decoded)
        autoencoder.compile(optimizer=Adam(), loss='mean_squared_error')
        return autoencoder

    def train_autoencoder(self, epochs: int = 10, batch_size: int = 128) -> History:
        """
        Train the Autoencoder on the reduced SVD matrix.

        Params:
        - epochs (int): Number of epochs for training (default is 50).
        - batch_size (int): Batch size for training (default is 256).

        Returns:
        - History: Training history object.
        """
        input_dim = self.svd_matrix.shape[1]
        self.autoencoder = self._build_autoencoder(input_dim, self.encoding_dim)
        history = self.autoencoder.fit(self.svd_matrix, self.svd_matrix, 
                                       epochs=epochs, batch_size=batch_size, 
                                       shuffle=True, verbose=1)
        encoder = Model(inputs=self.autoencoder.input, outputs=self.autoencoder.layers[1].output)
        self.encoded_matrix = encoder.predict(self.svd_matrix)
        self.cosine_sim = self._compute_cosine_similarity()
        return history

    def _compute_cosine_similarity(self) -> np.ndarray:
        """
        Compute the cosine similarity matrix from the encoded matrix.

        Returns:
        - np.ndarray: A matrix of cosine similarity scores.
        """
        return cosine_similarity(self.encoded_matrix, self.encoded_matrix)

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

    def save_model(self, model_path: str) -> None:
        """
        Save the trained autoencoder model.

        Params:
        - model_path (str): Path to save the trained model.
        """
        if self.autoencoder:
            self.autoencoder.save(model_path)

    def load_model(self, model_path: str) -> None:
        """
        Load a trained autoencoder model.

        Params:
        - model_path (str): Path to load the trained model from.
        """
        self.autoencoder = load_model(model_path)
        encoder = Model(inputs=self.autoencoder.input, outputs=self.autoencoder.layers[1].output)
        self.encoded_matrix = encoder.predict(self.svd_matrix)
        self.cosine_sim = self._compute_cosine_similarity()