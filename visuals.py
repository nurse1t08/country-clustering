import matplotlib.pyplot as plt
import seaborn as sns

def plot_correlation_heatmap(df):
    """Построение тепловой карты корреляций."""
    features = ['child_mort', 'exports', 'health', 'imports', 'income', 
                'inflation', 'life_expec', 'total_fer', 'gdpp']
    # Оставляем только те признаки, которые есть в датафрейме
    existing_features = [f for f in features if f in df.columns]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df[existing_features].corr(), annot=True, cmap='RdYlGn', fmt=".2f", ax=ax)
    plt.title("Корреляция признаков")
    return fig

def plot_distributions(df):
    """Графики распределения для каждого признака."""
    features = ['child_mort', 'exports', 'health', 'imports', 'income', 
                'inflation', 'life_expec', 'total_fer', 'gdpp']
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    for i, col in enumerate(features):
        if col in df.columns:
            sns.histplot(df[col], kde=True, ax=axes[i], color='skyblue')
            axes[i].set_title(f'Распределение: {col}')
    plt.tight_layout()
    return fig

def plot_clusters(df):
    """Визуализация кластеров на основе PCA."""
    fig, ax = plt.subplots(figsize=(10, 7))
    if 'pca1' in df.columns and 'pca2' in df.columns:
        sns.scatterplot(
            data=df, x='pca1', y='pca2', hue='cluster', 
            palette='viridis', s=100, alpha=0.7, ax=ax
        )
        plt.title("Визуализация кластеров (PCA)")
        plt.legend(title='Кластер')
    return fig
