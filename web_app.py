import streamlit as st
import pandas as pd
from pipeline import CountryClusteringPipeline
import visuals as vs
import os

st.set_page_config(page_title="Анализ Стран", layout="wide")

# Пути в корне папки archive
DATA_PATH = "/Users/aizat/Downloads/archive/clustered_countries.csv"
MODEL_DIR = "/Users/aizat/Downloads/archive/models/"

st.title("🌍 Система Кластеризации Стран")

if not os.path.exists(DATA_PATH):
    st.error("Ошибка: Сначала запустите 'python3 train.py' в терминале!")
else:
    df = pd.read_csv(DATA_PATH)
    
    st.subheader("📋 Данные (первые 10 строк)")
    st.dataframe(df.head(10))

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🔥 Корреляция")
        st.pyplot(vs.plot_correlation_heatmap(df))
    with col2:
        st.subheader("📍 Визуализация кластеров")
        st.pyplot(vs.plot_clusters(df))

    st.subheader("📊 Распределение показателей")
    st.pyplot(vs.plot_distributions(df))

    st.divider()
    st.subheader("🔮 Определить кластер для новой страны")
    
    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)
        f_inputs = {}
        features = ['child_mort', 'exports', 'health', 'imports', 'income', 
                    'inflation', 'life_expec', 'total_fer', 'gdpp']
        
        for i, f in enumerate(features):
            if i < 3: target_col = c1
            elif i < 6: target_col = c2
            else: target_col = c3
            f_inputs[f] = target_col.number_input(f, value=float(df[f].median()))
            
        submit = st.form_submit_button("Рассчитать")
        
        if submit:
            pipeline = CountryClusteringPipeline()
            cluster = pipeline.predict_new(f_inputs, path=MODEL_DIR)
            
            st.success(f"Результат: Кластер {cluster}")
            st.info("0 - Развитые, 1 - Бедные, 2 - Средние")
