import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import time

# =========================
# CONFIGURACIÓN DE PÁGINA
# =========================
st.set_page_config(
    page_title="Fashion MNIST Classifier",
    page_icon="👕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# ESTILOS CSS PERSONALIZADOS
# =========================
st.markdown("""
    <style>
    /* Estilo general */
    .main {
        padding: 2rem;
    }
    
    /* Título principal */
    .main-title {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .main-title h1 {
        font-size: 3rem;
        margin: 0;
        font-weight: bold;
    }
    
    .main-title p {
        font-size: 1.2rem;
        margin: 0.5rem 0 0 0;
        opacity: 0.95;
    }
    
    /* Tarjeta de predicción */
    .prediction-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        margin-top: 1rem;
    }
    
    .prediction-label {
        font-size: 1.2rem;
        color: #667eea;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    
    .prediction-value {
        font-size: 2.5rem;
        color: #764ba2;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    
    .confidence-value {
        font-size: 1.8rem;
        color: #27ae60;
        font-weight: bold;
    }
    
    /* Barra de progreso personalizada */
    .custom-progress {
        background-color: #e0e0e0;
        border-radius: 10px;
        padding: 3px;
        margin: 10px 0;
    }
    
    .custom-progress-bar {
        background: linear-gradient(90deg, #667eea, #764ba2);
        border-radius: 8px;
        height: 30px;
        display: flex;
        align-items: center;
        justify-content: flex-end;
        padding-right: 10px;
        color: white;
        font-weight: bold;
        transition: width 1s ease-in-out;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        margin-top: 3rem;
        color: #666;
        border-top: 1px solid #e0e0e0;
    }
    
    /* Info box */
    .info-box {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    /* Grid de categorías */
    .categories-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 8px;
        margin-top: 10px;
    }
    
    .category-item {
        background: #f8f9fa;
        padding: 5px 10px;
        border-radius: 8px;
        font-size: 0.9rem;
    }
    </style>
""", unsafe_allow_html=True)

# =========================
# CLASES COMPLETAS DE FASHION MNIST
# =========================
classes = [
    "Camiseta",
    "Pantalón",
    "Suéter",
    "Vestido",
    "Abrigo",
    "Sandalia",
    "Camisa",
    "Zapatilla",
    "Bolso",
    "Botín"
]

# Diccionario con emojis para cada prenda
class_emojis = {
    "Camiseta": "👕",
    "Pantalón": "👖",
    "Suéter": "🧥",
    "Vestido": "👗",
    "Abrigo": "🧥",
    "Sandalia": "👡",
    "Camisa": "👔",
    "Zapatilla": "👟",
    "Bolso": "👜",
    "Botín": "👢"
}

# =========================
# ENCABEZADO PERSONALIZADO
# =========================
st.markdown("""
    <div class="main-title">
        <h1>👕 Clasificador de Moda Fashion MNIST</h1>
        <p>Inteligencia Artificial para reconocer 10 tipos diferentes de prendas y accesorios</p>
    </div>
""", unsafe_allow_html=True)

# =========================
# SIDEBAR CON INFORMACIÓN
# =========================
with st.sidebar:
    st.markdown("### 📊 Información del Modelo")
    st.info("""
    **Modelo:** CNN (Red Neuronal Convolucional)
    **Dataset:** Fashion MNIST
    **Precisión:** ~92%
    **Clases:** 10 categorías de moda
    """)
    
    st.markdown("---")
    st.markdown("### 🎯 Todas las categorías")
    st.markdown("*(10 prendas y accesorios)*")
    
    # Mostrar TODAS las 10 categorías en el sidebar
    for clase in classes:
        emoji = class_emojis.get(clase, "👕")
        st.markdown(f"- {emoji} **{clase}**")
    
    st.markdown("---")
    st.markdown("### ℹ️ Instrucciones")
    st.markdown("""
    1. 📤 Sube una imagen (JPG, PNG)
    2. ⏳ Espera el procesamiento
    3. ✨ Obtén la predicción
    """)
    
    st.markdown("---")
    st.markdown("### 💡 Consejo")
    st.markdown("""
    Para mejores resultados, usa:
    - ✅ Fondo claro
    - ✅ Prenda centrada
    - ✅ Buen contraste
    """)

# =========================
# FUNCIONES
# =========================
@st.cache_resource
def cargar_modelo():
    # Apunta exactamente al nombre de tu archivo (respetando minúsculas)
    return load_model("modelo.keras")

model = cargar_modelo()

# Función para preprocesar imagen
def preprocesar_imagen(image):
    # Convertir a escala de grises
    image = image.convert("L")
    # Redimensionar a 28x28
    image = image.resize((28, 28))
    # Convertir a array
    img = np.array(image)
    # Invertir colores (Fashion MNIST tiene fondo negro)
    img = 255 - img
    # Normalizar
    img = img / 255.0
    # Redimensionar para la entrada de la CNN (1, 28, 28, 1)
    img = img.reshape(1, 28, 28, 1)
    return img

# =========================
# ÁREA PRINCIPAL
# =========================
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    # Uploader de archivos con diseño mejorado
    uploaded_file = st.file_uploader(
        "### 📤 Sube tu imagen aquí",
        type=["jpg", "jpeg", "png"],
        help="Formatos soportados: JPG, JPEG, PNG"
    )

# =========================
# PREDICCIÓN
# =========================
if uploaded_file is not None:
    # Crear columnas para mostrar imagen y resultado
    col_imagen, col_resultado = st.columns(2)
    
    with col_imagen:
        st.markdown("### 📷 Imagen subida")
        image = Image.open(uploaded_file)
        
        # Mostrar imagen con marco
        st.image(image, width=300, use_container_width=True)
        
        # Mostrar información de la imagen
        st.caption(f"Tamaño original: {image.size[0]}x{image.size[1]} píxeles")
    
    with col_resultado:
        st.markdown("### 🔍 Procesando...")
        
        # Barra de progreso animada
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Simular procesamiento
        for i in range(100):
            progress_bar.progress(i + 1)
            if i < 30:
                status_text.text("🔄 Preprocesando imagen...")
            elif i < 60:
                status_text.text("🧠 Analizando con IA...")
            elif i < 90:
                status_text.text("📊 Calculando probabilidades...")
            else:
                status_text.text("✅ ¡Completado!")
            time.sleep(0.01)
        
        # Preprocesar y predecir
        img_procesada = preprocesar_imagen(image)
        pred = model.predict(img_procesada, verbose=0)
        clase_idx = np.argmax(pred)
        confianza = np.max(pred) * 100
        
        # Limpiar elementos de progreso
        progress_bar.empty()
        status_text.empty()
        
        # Mostrar resultado con diseño mejorado
        st.markdown("### ✨ Resultado de la predicción")
        
        # Obtener emoji de la clase predicha
        clase_predicha = classes[clase_idx]
        emoji_predicho = class_emojis.get(clase_predicha, "👕")
        
        # Tarjeta de resultado
        st.markdown(f"""
        <div class="prediction-card">
            <div class="prediction-label">Prenda identificada</div>
            <div class="prediction-value">{emoji_predicho} {clase_predicha}</div>
            <div class="prediction-label">Nivel de confianza</div>
            <div class="confidence-value">{confianza:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Barra de confianza personalizada
        st.markdown("### 📈 Top 3 predicciones")
        
        # Mostrar top 3 predicciones
        top_indices = np.argsort(pred[0])[-3:][::-1]
        for idx in top_indices:
            proba = pred[0][idx] * 100
            clase_name = classes[idx]
            emoji = class_emojis.get(clase_name, "👕")
            st.markdown(f"""
            <div style="margin: 10px 0;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                    <span style="font-weight: bold;">{emoji} {clase_name}</span>
                    <span style="color: #666;">{proba:.1f}%</span>
                </div>
                <div class="custom-progress">
                    <div class="custom-progress-bar" style="width: {proba}%;">
                        {proba:.1f}%
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# =========================
# SECCIÓN DE INFORMACIÓN (cuando no hay imagen)
# =========================
if not uploaded_file:
    with col2:
        st.markdown("""
        <div class="info-box">
        💡 <strong>¿Cómo funciona?</strong><br>
        1. Sube una imagen de una prenda o accesorio<br>
        2. La IA procesará y analizará la imagen<br>
        3. Recibirás una predicción con nivel de confianza<br>
        </div>
        """, unsafe_allow_html=True)
        
        # Mostrar todas las categorías que puede identificar
        st.markdown("### 🎯 Puedo identificar estas 10 prendas:")
        
        # Crear grid de 2 columnas para mostrar las 10 categorías
        cols_cat = st.columns(2)
        for i, clase in enumerate(classes):
            emoji = class_emojis.get(clase, "👕")
            with cols_cat[i % 2]:
                st.markdown(f"{emoji} **{clase}**")
        
        st.markdown("---")
        st.markdown("### 🖼️ ¿Qué tipo de imágenes funcionan mejor?")
        col_ej1, col_ej2, col_ej3 = st.columns(3)
        with col_ej1:
            st.markdown("✅ **Buen ejemplo**\n\nFondo claro\nPrenda centrada\nBuen contraste")
        with col_ej2:
            st.markdown("⚠️ **Ejemplo regular**\n\nFondo oscuro\nPrenda pequeña\nPoca luz")
        with col_ej3:
            st.markdown("❌ **Mal ejemplo**\n\nMúltiples objetos\nFondo desordenado\nImagen borrosa")

# =========================
# FOOTER
# =========================
st.markdown("""
    <div class="footer">
        <p>🤖 Modelo entrenado con Fashion MNIST | 📊 10 categorías de moda | 🚀 Desplegado con Streamlit</p>
        <p style="font-size: 0.8rem;">Categorías: Camiseta, Pantalón, Suéter, Vestido, Abrigo, Sandalia, Camisa, Zapatilla, Bolso, Botín</p>
    </div>
""", unsafe_allow_html=True)