import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import os

class CountryClusteringPipeline:
    def __init__(self, n_clusters=3):
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=2)
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.features = ['child_mort', 'exports', 'health', 'imports', 'income', 
                         'inflation', 'life_expec', 'total_fer', 'gdpp']
        # Базовый путь — текущая папка (archive)
        self.base_path = os.path.dirname(os.path.abspath(__file__))

    def prepare_data(self, df):
        """Масштабирование числовых признаков."""
        X = df[self.features]
        X_scaled = self.scaler.fit_transform(X)
        return X_scaled

    def train(self, df):
        """Обучение модели и PCA."""
        X_scaled = self.prepare_data(df)
        clusters = self.kmeans.fit_predict(X_scaled)
        pca_data = self.pca.fit_transform(X_scaled)
        
        df['cluster'] = clusters
        df['pca1'] = pca_data[:, 0]
        df['pca2'] = pca_data[:, 1]
        return df

    def save_models(self, path=None):
        """Сохранение моделей в папку models."""
        if path is None:
            path = os.path.join(self.base_path, 'models')
        if not os.path.exists(path):
            os.makedirs(path)
        joblib.dump(self.kmeans, os.path.join(path, 'kmeans_model.joblib'))
        joblib.dump(self.scaler, os.path.join(path, 'scaler.joblib'))
        joblib.dump(self.pca, os.path.join(path, 'pca_transformer.joblib'))
        print(f"Модели успешно сохранены в: {path}")

    def predict_new(self, data_dict, path=None):
        """Предсказание для новых данных."""
        if path is None:
            path = os.path.join(self.base_path, 'models')
            
        kmeans = joblib.load(os.path.join(path, 'kmeans_model.joblib'))
        scaler = joblib.load(os.path.join(path, 'scaler.joblib'))
        
        df_input = pd.DataFrame([data_dict])
        X_scaled = scaler.transform(df_input[self.features])
        cluster = kmeans.predict(X_scaled)
        return cluster[0]
