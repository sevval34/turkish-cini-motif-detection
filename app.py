import streamlit as st
from ultralytics import YOLO
from PIL import Image

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="Çini Motiflerinin Sessiz Anlatısı",
    layout="wide"
)

# ==================================================
# GLOBAL CSS
# ==================================================
st.markdown("""
<style>
.stApp {
    background-color: #FBF8F3;
    color: #1f2937;
}

/* NAVBAR */
.navbar {
    display: flex;
    justify-content: center;
    gap: 28px;
    padding: 18px 0;
    border-bottom: 1px solid #E5E7EB;
}

.nav-icon {
    font-size: 1.3rem;
    color: #0B2545;
}

/* HERO CARD */
.hero-card {
    max-width: 820px;
    margin: 50px auto 40px auto;
    padding: 45px;
    background-color: #ffffff;
    border-radius: 16px;
    border: 1px solid #E5E7EB;
    text-align: center;
}

/* TITLE */
.title {
    font-size: 3rem;
    font-weight: 700;
    color: #0B2545;
    letter-spacing: 1px;
}

/* SUBTITLE */
.subtitle {
    font-size: 1.15rem;
    color: #475569;
    margin-top: 12px;
}

/* BUTTON */
.stButton > button {
    background-color: #0B2545;
    color: #FBF8F3;
    border-radius: 8px;
    padding: 10px 24px;
    font-weight: 500;
}
.stButton > button:hover {
    background-color: #133B6F;
}

/* RESULT CARD */
.result-card {
    max-width: 900px;
    margin: 40px auto;
    padding: 40px;
    background-color: #ffffff;
    border-radius: 14px;
    border: 1px solid #E5E7EB;
}

.motif-title {
    font-size: 1.5rem;
    font-weight: 600;
    color: #0B2545;
    margin-top: 30px;
}

.short {
    font-weight: 500;
    margin-top: 8px;
}

.mid {
    color: #374151;
    margin-top: 8px;
    line-height: 1.6;
}

/* FOOTER */
.footer {
    margin-top: 90px;
    padding: 50px 0;
    border-top: 1px solid #E5E7EB;
    color: #475569;
    font-size: 0.95rem;
}
</style>
""", unsafe_allow_html=True)

# ==================================================
# NAVBAR (MINIMAL ICONS)
# ==================================================
st.markdown("""
<div class="navbar">
    <div class="nav-icon">❁</div>
    <div class="nav-icon">● ● ●</div>
    <div class="nav-icon">❀</div>
    <div class="nav-icon">✿</div>
</div>
""", unsafe_allow_html=True)

# ==================================================
# HERO SECTION (TITLE + SUBTITLE IN SAME CARD)
# ==================================================
st.markdown("""
<div class="hero-card">
    <div class="title">Çini Motiflerinin Sessiz Anlatısı</div>
    <div class="subtitle">
        Geleneksel sembolleri yapay zekâ ile keşfedin
    </div>
</div>
""", unsafe_allow_html=True)

# ==================================================
# MODEL
# ==================================================
MODEL_PATH = "runs/detect/train/weights/best.pt"

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

try:
    model = load_model()
except:
    st.error("Model yüklenemedi. Model yolu kontrol edilmelidir.")
    st.stop()

# ==================================================
# MOTIF DATA (LAYERED)
# ==================================================
motifs = {
    "lale": {
        "kisa": "Bu motif laledir. Osmanlı çini sanatının en güçlü sembollerinden biridir.",
        "orta": "Lale, tasavvufta Allah’ın birliğini ve zarafeti temsil eder.",
        "uzun": "İznik çinilerinde yoğun olarak kullanılmış, mistik anlamlar kazanmıştır."
    },
    "karanfil": {
        "kisa": "Bu motif karanfildir.",
        "orta": "Bereket ve sürekliliği simgeler.",
        "uzun": "Osmanlı süsleme sanatında doğayla ilişkilidir."
    },
    "cintemani": {
        "kisa": "Bu motif çintemanidir.",
        "orta": "Güç ve hâkimiyet sembolüdür.",
        "uzun": "Orta Asya kökenlidir, saray sanatında yaygındır."
    }
}

# ==================================================
# UPLOAD AREA
# ==================================================
uploaded_file = st.file_uploader(
    "Görsel yükleyin, yapay zeka çini motiflerini ve sembolik anlamlarını analiz etsin",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file)

    # Görsel boyut KONTROLLÜ
    st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)
    st.image(image, caption="Yüklenen Görsel", width=550)
    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("Analiz Et"):
        with st.spinner("Motifler analiz ediliyor..."):
            results = model.predict(source=image, conf=0.4)
            detected = set()

            for box in results[0].boxes:
                detected.add(model.names[int(box.cls[0])])

            if detected:
                st.markdown('<div class="result-card">', unsafe_allow_html=True)

                for d in detected:
                    info = motifs.get(d.lower())
                    if info:
                        st.markdown(f'<div class="motif-title">{d.upper()}</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="short">{info["kisa"]}</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="mid">{info["orta"]}</div>', unsafe_allow_html=True)

                        with st.expander("Daha fazla bilgi"):
                            st.write(info["uzun"])

                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.warning("Tanımlı bir motif tespit edilemedi.")

# ==================================================
# FOOTER
# ==================================================
st.markdown("""
<div class="footer">
    <div style="max-width:1000px; margin:auto; display:flex; justify-content:space-between;">
        <div style="width:45%;">
            <b>Hakkımızda</b><br><br>
            Bu proje, çini sanatındaki motiflerin yapay zekâ ile tanınmasını ve
            kültürel anlamlarının dijital ortamda aktarılmasını amaçlamaktadır.
        </div>
        <div style="width:45%;">
            <b>Ekibimiz</b><br><br>
            Nisa Çibik<br>
            Melike Arslan<br>
            Şevval Başkan<br>
            Berkce Cebeci<br>
            Begüm Alakuş
        </div>
    </div>
</div>
""", unsafe_allow_html=True)
