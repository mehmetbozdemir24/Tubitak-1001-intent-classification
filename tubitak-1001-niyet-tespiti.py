import os
import json
import random
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import streamlit as st
from google import genai

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

# Sayfa Yapılandırması
st.set_page_config(
    page_title="Intent & Alignment Studio",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Session State Tanımlamaları (arayüz durum rozeti bunlara erişeceği için CSS'ten önce tanımlanır)
if "model" not in st.session_state:
    st.session_state.model = None
if "tokenizer" not in st.session_state:
    st.session_state.tokenizer = None
if "label_encoder" not in st.session_state:
    st.session_state.label_encoder = None
if "eval_results" not in st.session_state:
    st.session_state.eval_results = None
if "cm" not in st.session_state:
    st.session_state.cm = None
if "classes" not in st.session_state:
    st.session_state.classes = None
if "dpo_dataset_buffer" not in st.session_state:
    st.session_state.dpo_dataset_buffer = []

# DPO verilerinin diske yazılacağı sabit dosya yolu
DPO_OUTPUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dpo_dataset.jsonl")

# ------------------------------------------------------------------
# DPO ÜRETİM PARAMETRELERİ — Gemini'ye gidecek talimatın kalitesini
# artırmak için sınıf tanımları, çeşitlilik ve yasak kelime listeleri.
# ------------------------------------------------------------------

# Her sınıfın kısa tanımı + örnekleri + karıştırılmaması gereken sınıfla farkı.
# Gemini talimatına hem hedef sınıfın TAM tanımı hem de çeldirici sınıfın
# KISA tanımı eklenir ki model "asıl istek" ile "çeldirici" ayrımını net görsün.
SINIF_TANIMLARI = {
    "Bilgi_Karti": {
        "tanim": "Öğrenci konuyu tekrar/ezber için AYRI AYRI KARTLAR (flashcard, soru-cevap kartı, ön yüz/arka yüz) istiyor. Not: 'soru-cevap kartı' Bilgi_Karti'dır, Kavramsal_Soru değil.",
        "ornekler": [
            "Enzimlerin işleyişini kart destesine dökebilir misin?",
            "hal değişimi grafikleri aklımda hiç kalmıyor, birkaç tekrar kartı çıkarsana",
            "endositoz ekzositoz kartı",
        ],
        "fark": "Anahtar_Kavram_Ozet tek parça toplu metin ister; kart ayrı ayrı tekrar birimleri ister.",
    },
    "Anahtar_Kavram_Ozet": {
        "tanim": "Konunun ÖZETİNİ, ana hatlarını, önemli kavramlarını toplu/madde madde istiyor.",
        "ornekler": [
            "Aristoteles'in mantıktaki kavram ve terim anlayışını özetle.",
            "moğol istilası konusu zihnimde dağınık kaldı, kısa toparlama lazım",
            "nüfus piramidi özeti",
        ],
        "fark": "Bilgi_Karti ayrı tekrar kartları ister; bu sınıf tek parça toplu özet ister.",
    },
    "Kavramsal_Soru": {
        "tanim": "Öğrenci KENDİSİNE anlayışını ölçen/düşündüren (tek) bir soru SORULMASINI istiyor (öğrencinin kendisinin soru sorması değil).",
        "ornekler": [
            "osmozu çalıştım, beni düşündürecek bi soru sorsana",
            "Şiir bilgisiyle ilgili kısa bir quiz sorusu at.",
            "türkiyedeki su kaynaklarını öğrendim, bilgimi merak ediyorum, dener misin",
        ],
        "fark": "Uyarlanabilir_Soru'da zorluk cevaplara göre ayarlanır; burada ayarlama yok.",
    },
    "Uyarlanabilir_Soru": {
        "tanim": "Kendisine sorular sorulmasını ve ZORLUĞUN cevaplarına/seviyesine göre AYARLANMASINI istiyor (kolaydan zora, bildikçe zorlaştır, takılırsam kolaylaştır).",
        "ornekler": [
            "hinduizmden sor, ilerledikçe zorlaştır",
            "asit baz tepkimelerinde takılırsam basit bir soruya dönülsün, rahatladıkça zorluk artsın.",
            "hocam stresin nedenlerinden sor, seviyemi ölçüp ona göre sor",
        ],
        "fark": "Kavramsal_Soru'da tek soru sorulur, zorluk ayarlaması yoktur.",
    },
    "Yanlis_Cevap_Aciklama": {
        "tanim": "Daha önce çözdüğü bir soruda YANLIŞ yaptığını söylüyor ve hatasının nerede/neden olduğunu soruyor.",
        "ornekler": [
            "Pisagor sorusunda cevabım 13 çıktı ama doğrusu 12'ymiş, nerede hata yaptığımı gösterir misin?",
            "suffix hatam nerede",
            "katı sıvı ile ilgili soruda bir özelliği ters yazmışım, kafam karıştı bu soruda",
        ],
        "fark": "Diğer sınıflarda öğrenci henüz bir soru çözmüş/yanlış yapmış olduğunu belirtmez.",
    },
    "Kapsam_Disi": {
        "tanim": "Selamlaşma, teşekkür, veda, sohbet, duygu/motivasyon, güncel olay/spor/film, uygulama hakkında soru, anlamsız metin — yukarıdaki 5 eğitim talebinden hiçbiri değil. Ders konusunun ADI geçse bile istek bu 5 talepten biri değilse (konunun tarihi, kitap/belgesel önerisi, kariyer, 'ne işime yarayacak', arkadaşa nasıl anlatırım) Kapsam_Disi'dır.",
        "ornekler": [
            "Selamünaleyküm, nasılsınız hocalarım?",
            "Türev çalışmaktan çok sıkıldım, bu hafta sonu ne izlesem?",
            "asdasd",
        ],
        "fark": "Diğer 5 sınıf her zaman somut bir ders/eğitim talebi içerir; bu sınıf içermez.",
    },
}

# Yasaklı ifadeler — toplanan verideki kalıplaşmış, aşırı tekrarlanan kelime/ifadeler.
# Kelime sınırlı (örn. "hap" "hapis" içinde eşleşmez) ve büyük/küçük harf duyarsız aranır.
YASAKLI_IFADELER = [
    "hap bilgi",
    "hap",
    "can alıcı",
    "yarınki deneme sınavı",
    "uzun uzadıya konu anlatımı",
]

# Normal kategori üretiminde rastgele seçilecek üslup çeşitleri
URETIM_USLUPLARI = [
    "Resmi ve saygılı bir dille yaz.",
    "Günlük, samimi bir sohbet diliyle yaz.",
    "'Hocam' diye hitap eden, samimi bir ton kullan.",
    "Yazım hataları olan, Türkçe karaktersiz (ı,ş,ğ,ç,ö,ü yerine i,s,g,c,o,u kullanan) bir öğrenci diliyle yaz.",
    "Çok kısa ve kesik cümlelerle yaz.",
    "Önce kısa bir bağlam/durum anlat, sonra isteği belirt.",
]

# "Aşırı Zorlayıcı" modunda kullanılacak çeldirici teknikleri
CELDIRICI_TEKNIKLERI = [
    "Başka bir sınıfın imza kelimesini olumsuzlayarak başla (örn. 'özet istemiyorum, ...'), sonra asıl isteği belirt.",
    "'{celdirici} yerine {hedef}' kalıbıyla asıl isteği netleştir.",
    "Geçmiş bir olay anlatarak başla (örn. 'dün ... yanlış yaptım ama ...'), sonra asıl isteğe geç.",
    "Asıl isteği bir sohbetin/durumun içine gömerek ilet.",
    "Hedef sınıfın imza kelimesini (örn. 'kart', 'özet' gibi) hiç kullanmadan, dolaylı biçimde iste.",
]

# Kapsam_Dışı "excel" modunda kullanılacak sabit kapsam dışı talepler
# (mevcut 10 madde + çeşitlilik için eklenen yeni maddeler)
KAPSAM_DISI_TALEPLERI = [
    "bu konuyu arkadaşıma nasıl anlatabilirim",
    "bu konuyla ilgili kitap önerir misin",
    "bu konuyu kimler icat etmiş",
    "bu konunun tarihi nedir",
    "bu konuyla ilgili film veya belgesel var mı",
    "bu konuyu öğrenmenin ne faydası var",
    "bu konuyu sevmiyorum ne yapmalıyım",
    "bu konuyla ilgili yarışmalar var mı",
    "bu konuyu gelecekte kullanacak mıyım",
    "bu konuda kariyer yapabilir miyim",
    "bu konuyu anlattığınız için teşekkür ederim, iyi günler",
    "bu konudan nefret ediyorum, çok sıkıcı geliyor diye şikayet ediyorum",
    "bu uygulamayı nasıl kullanacağımı soruyorum",
    "sınavda bu konudan kaç soru çıkar diye tahmin istiyorum",
    "bu konuyu çalışırken müzik dinlesem mi diye tavsiye istiyorum",
    "bu konuyla ilgili önerebileceğin bir YouTube kanalı var mı diye soruyorum",
]


def _sinif_kisa_tanim(sinif_adi):
    """Bir sınıfın SINIF_TANIMLARI içindeki kısa tanımını döndürür (bulunamazsa sınıf adını)."""
    bilgi = SINIF_TANIMLARI.get(sinif_adi)
    return bilgi["tanim"] if bilgi else sinif_adi


def _hedef_tanimini_formatla(hedef_niyet):
    """Hedef sınıfın TAM tanımını (tanım + örnekler + fark) talimat metni olarak döndürür."""
    bilgi = SINIF_TANIMLARI.get(hedef_niyet)
    if not bilgi:
        return f"HEDEF NİYET: '{hedef_niyet}' kategorisine ait bir talep yaz."
    ornekler_metni = "\n".join(f"  - \"{orn}\"" for orn in bilgi["ornekler"])
    return (
        f"HEDEF NİYET TANIMI — {hedef_niyet}:\n"
        f"{bilgi['tanim']}\n"
        f"Örnekler:\n{ornekler_metni}\n"
        f"Karıştırılmaması gereken fark: {bilgi['fark']}"
    )


def _celdirici_sinifi_formatla(celdirici_sinif):
    """Çeldirici sınıfın KISA tanımını talimat metni olarak döndürür."""
    return f"ÇELDİRİCİ SINIF (yalnızca yan unsur olarak kullan) — {celdirici_sinif}:\n{_sinif_kisa_tanim(celdirici_sinif)}"


def _yasakli_ifade_var_mi(metin):
    """Metinde YASAKLI_IFADELER listesindeki bir ifade kelime sınırlı olarak geçiyor mu kontrol eder."""
    for ifade in YASAKLI_IFADELER:
        kalip = r"(?<!\w)" + re.escape(ifade) + r"(?!\w)"
        if re.search(kalip, metin, re.IGNORECASE):
            return True
    return False


def _gemini_ciktisini_temizle(metin):
    """Gemini'den gelen metindeki baştaki/sondaki tırnak işaretlerini ve 'Öğrenci:' gibi önekleri temizler."""
    metin = metin.strip()
    metin = metin.strip("\"'“”‘’`")
    metin = re.sub(r"^(öğrenci|ogrenci|student)\s*[:：]\s*", "", metin, flags=re.IGNORECASE)
    return metin.strip()

# ------------------------------------------------------------------
# TASARIM SİSTEMİ — "Lab Console" tema
# Koyu, teknik bir enstrüman-paneli hissi: sinyal teali vurgu rengi,
# DPO'nun ikili doğasını (chosen/rejected) kodlayan zümrüt/turuncu ikili.
# ------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600&display=swap');

    :root {
        --bg: #F5F9FD;
        --surface: #EAF2FA;
        --surface-2: #DCEAF7;
        --border: #C7DEF2;
        --border-strong: #A9CBEB;
        --text: #10151C;
        --text-muted: #4B5A6B;
        --text-faint: #8AA0B5;
        --accent: #1D6FD8;
        --accent-strong: #1558AE;
        --accent-soft: rgba(29, 111, 216, 0.10);
        --chosen: #15803D;
        --rejected: #C2670C;
        --danger: #C0293D;
        --radius: 10px;
    }

    html, body, [data-testid="stAppViewContainer"], .main { background-color: var(--bg) !important; }
    [data-testid="stHeader"] { background-color: transparent !important; }
    [data-testid="stSidebar"] { background-color: var(--surface) !important; border-right: 1px solid var(--border); }

    * { font-family: 'Inter', sans-serif; }
    h1, h2, h3, h4, h5 { font-family: 'Space Grotesk', sans-serif !important; color: var(--text) !important; letter-spacing: -0.01em; }
    p, span, label, div { color: var(--text); }
    code, .mono { font-family: 'JetBrains Mono', monospace !important; }

    /* Sekme içerik alanlarına hafif nefes payı */
    .stTabs [data-baseweb="tab-panel"] { padding-top: 20px; }

    /* Durum rozeti */
    .status-badge { display: inline-flex; align-items: center; gap: 7px; font-family: 'JetBrains Mono', monospace; font-size: 12px; padding: 5px 12px; border-radius: 999px; border: 1px solid var(--border-strong); color: var(--text-muted); }
    .status-dot { width: 7px; height: 7px; border-radius: 50%; background: var(--text-faint); }
    .status-dot.on { background: var(--accent); box-shadow: 0 0 6px var(--accent); }

    /* Metric rozetleri */
    .metric-badge { background-color: var(--accent-soft); color: var(--accent); padding: 6px 14px; border-radius: 999px; font-weight: 600; font-family: 'JetBrains Mono', monospace; font-size: 13px; display: inline-block; border: 1px solid rgba(29, 111, 216, 0.25); }
    .tag-chosen { color: var(--chosen); border: 1px solid rgba(21, 128, 61, 0.3); background: rgba(21, 128, 61, 0.08); padding: 4px 10px; border-radius: 6px; font-family: 'JetBrains Mono', monospace; font-size: 12px; font-weight: 600; }
    .tag-rejected { color: var(--rejected); border: 1px solid rgba(194, 103, 12, 0.3); background: rgba(194, 103, 12, 0.08); padding: 4px 10px; border-radius: 6px; font-family: 'JetBrains Mono', monospace; font-size: 12px; font-weight: 600; }

    /* Olasılık çubukları (signal meter) */
    .prob-row { display: flex; align-items: center; gap: 12px; margin: 7px 0; }
    .prob-label { width: 30%; font-size: 13px; color: var(--text-muted); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .prob-track { flex: 1; height: 8px; background: var(--surface-2); border-radius: 4px; overflow: hidden; border: 1px solid var(--border); }
    .prob-fill { height: 100%; border-radius: 4px; background: var(--accent); }
    .prob-fill.top { background: linear-gradient(90deg, var(--accent-strong), var(--accent)); }
    .prob-value { width: 52px; text-align: right; font-family: 'JetBrains Mono', monospace; font-size: 12px; color: var(--text); }

    /* Streamlit bileşen override'ları */
    .stButton>button, .stDownloadButton>button {
        border-radius: 7px; font-weight: 600; font-family: 'Inter', sans-serif;
        border: 1px solid var(--border-strong); transition: all 0.15s ease;
        background-color: var(--surface-2); color: var(--text);
    }
    .stButton>button:hover, .stDownloadButton>button:hover { border-color: var(--accent); color: var(--accent); }
    .stButton>button[kind="primary"], .stDownloadButton>button[kind="primary"] {
        background-color: var(--accent); border-color: var(--accent); color: #FFFFFF;
    }
    .stButton>button[kind="primary"]:hover, .stDownloadButton>button[kind="primary"]:hover {
        background-color: var(--accent-strong); color: #FFFFFF;
    }

    .stTextInput input, .stTextArea textarea, .stNumberInput input {
        border-radius: 7px !important; border: 1px solid var(--border-strong) !important;
        background-color: var(--surface-2) !important; color: var(--text) !important;
        font-family: 'JetBrains Mono', monospace !important; font-size: 13.5px !important;
        -webkit-text-fill-color: var(--text) !important;
    }
    .stTextInput input::placeholder, .stTextArea textarea::placeholder { color: var(--text-faint) !important; opacity: 1 !important; }

    .stSelectbox [data-baseweb="select"] > div {
        border-radius: 7px !important; border: 1px solid var(--border-strong) !important;
        background-color: var(--surface-2) !important; color: var(--text) !important;
    }
    .stSelectbox [data-baseweb="select"] * { color: var(--text) !important; }

    /* Açılır seçenek listesi (portal olarak body sonuna render edilir) */
    div[data-baseweb="popover"] { background-color: var(--surface) !important; border: 1px solid var(--border-strong) !important; border-radius: 8px !important; box-shadow: 0 8px 24px rgba(16, 21, 28, 0.12) !important; }
    div[data-baseweb="popover"] ul, div[role="listbox"], div[role="listbox"] ul, ul[role="listbox"] { background-color: var(--surface) !important; }
    div[data-baseweb="popover"] li, div[role="listbox"] li, ul[role="listbox"] li, div[data-baseweb="popover"] [role="option"] {
        background-color: var(--surface) !important; color: var(--text) !important;
    }
    div[data-baseweb="popover"] li:hover, div[role="listbox"] li:hover,
    div[data-baseweb="popover"] [aria-selected="true"], div[role="listbox"] [aria-selected="true"] {
        background-color: var(--accent-soft) !important; color: var(--accent) !important;
    }

    /* Sekmeler -> segmented pipeline nav */
    .stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: 1px solid var(--border); }
    .stTabs [data-baseweb="tab"] {
        height: 44px; background-color: transparent; border: 1px solid transparent;
        border-radius: 7px 7px 0 0; font-family: 'JetBrains Mono', monospace;
        font-size: 13px; color: var(--text-muted); padding: 0 18px;
    }
    .stTabs [aria-selected="true"] {
        color: var(--accent) !important; background-color: var(--accent-soft);
        border: 1px solid var(--border); border-bottom: 2px solid var(--accent);
    }
    .stTabs [data-baseweb="tab-highlight"] { background-color: transparent; }

    [data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace !important; color: var(--accent) !important; font-size: 1.7rem !important; }
    [data-testid="stMetricLabel"] { color: var(--text-muted) !important; font-size: 12.5px !important; text-transform: uppercase; letter-spacing: 0.04em; }

    .stProgress > div > div > div { background-color: var(--accent) !important; }
    .stProgress > div > div { background-color: var(--surface-2) !important; }

    [data-testid="stFileUploaderDropzone"] { background-color: var(--surface-2) !important; border: 1px dashed var(--border-strong) !important; border-radius: var(--radius) !important; }

    /* Bildirim kutuları (info/success/warning/error) — Streamlit sürümüne göre 2 farklı isimlendirmeyi de kapsar */
    .stAlert, [data-testid="stAlertContainer"], [data-testid^="stNotification"] {
        border-radius: 8px !important; border: 1px solid var(--border-strong) !important;
        background-color: var(--surface-2) !important;
    }
    .stAlert *, [data-testid="stAlertContainer"] *, [data-testid^="stNotification"] * { color: var(--text) !important; }
    .stAlert [data-testid*="Success"], [data-testid="stAlertContainer"]:has(svg[data-testid*="Success"]) * ,
    [data-testid="stNotificationContentSuccess"] { color: var(--chosen) !important; }
    .stAlert [data-testid*="Warning"], [data-testid="stAlertContainer"]:has(svg[data-testid*="Warning"]) *,
    [data-testid="stNotificationContentWarning"] { color: var(--rejected) !important; }
    .stAlert [data-testid*="Error"], [data-testid="stAlertContainer"]:has(svg[data-testid*="Error"]) *,
    [data-testid="stNotificationContentError"] { color: var(--danger) !important; }
    .stAlert [data-testid*="Info"], [data-testid="stAlertContainer"]:has(svg[data-testid*="Info"]) *,
    [data-testid="stNotificationContentInfo"] { color: var(--accent) !important; }

    hr { border-color: var(--border) !important; }
    </style>
""",
    unsafe_allow_html=True,
)

# Başlık Bölümü
model_hazir = st.session_state.model is not None
st.markdown(
    f"""
    <div style='padding: 6px 0 20px 0; display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 12px;'>
        <div>
            <h1 style='margin-bottom: 4px; font-size: 2rem;'>Intent &amp; Alignment Studio</h1>
            <p style='color: var(--text-muted); font-size: 0.98rem; max-width: 640px;'>BERT niyet sınıflandırma modeli eğitimi, performans denetimi ve insan denetimli DPO hizalama verisi toplama hattı.</p>
        </div>
        <div class='status-badge'>
            <span class='status-dot {"on" if model_hazir else ""}'></span>
            {"MODEL YÜKLÜ" if model_hazir else "MODEL BEKLENIYOR"}
        </div>
    </div>
""",
    unsafe_allow_html=True,
)

# --- SEKMELER (TABS) ---
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "01 · VERİ & MODEL",
        "02 · PERFORMANS",
        "03 · MANUEL TEST",
        "04 · DPO TOPLAMA",
    ]
)

with tab1:
    with st.container():
        st.header("Model ve Veri İşlemleri")
        islem_turu = st.radio(
            "Çalışma Modu Seçin:",
            ["Yeni Model Eğit (Train)", "Daha Önce Kaydedilmiş Modeli Yükle (Load)"],
            horizontal=True,
        )

        if islem_turu == "Yeni Model Eğit (Train)":
            uploaded_file = st.file_uploader(
                "JSONL Formatında Veri Setinizi Yükleyin", type=["jsonl"]
            )
            col1, col2 = st.columns(2)
            with col1:
                model_name = st.selectbox(
                    "Model Seçimi", ["dbmdz/bert-base-turkish-cased"]
                )
                epochs = st.number_input(
                    "Epoch Sayısı", min_value=1, max_value=10, value=3
                )
            with col2:
                batch_size = st.selectbox("Batch Size", [8, 16, 32], index=1)
                learning_rate = st.number_input(
                    "Learning Rate", value=2e-5, format="%.5f"
                )

            if uploaded_file is not None:
                with open("temp_dataset.jsonl", "wb") as f:
                    f.write(uploaded_file.getbuffer())

                if st.button(
                    "Modeli Eğitmeye Başla", type="primary", use_container_width=True
                ):
                    with st.spinner("Model optimize ediliyor ve eğitiliyor..."):
                        data = []
                        with open("temp_dataset.jsonl", "r", encoding="utf-8") as f:
                            for line in f:
                                data.append(json.loads(line.strip()))

                        df = pd.DataFrame(data)

                        label_encoder = LabelEncoder()
                        df["label_id"] = label_encoder.fit_transform(df["label"])
                        num_labels = len(label_encoder.classes_)

                        train_texts, test_texts, train_labels, test_labels = (
                            train_test_split(
                                df["text"].tolist(),
                                df["label_id"].tolist(),
                                test_size=0.2,
                                random_state=42,
                                stratify=df["label_id"],
                            )
                        )

                        dataset = DatasetDict(
                            {
                                "train": Dataset.from_dict(
                                    {"text": train_texts, "label": train_labels}
                                ),
                                "test": Dataset.from_dict(
                                    {"text": test_texts, "label": test_labels}
                                ),
                            }
                        )

                        tokenizer = AutoTokenizer.from_pretrained(model_name)

                        def tokenize_function(examples):
                            return tokenizer(
                                examples["text"],
                                padding="max_length",
                                truncation=True,
                                max_length=128,
                            )

                        tokenized_datasets = dataset.map(
                            tokenize_function, batched=True
                        )
                        model = AutoModelForSequenceClassification.from_pretrained(
                            model_name, num_labels=num_labels
                        )

                        def compute_metrics(eval_pred):
                            logits, labels = eval_pred
                            preds = np.argmax(logits, axis=1)
                            precision, recall, f1, _ = precision_recall_fscore_support(
                                labels, preds, average="weighted"
                            )
                            acc = accuracy_score(labels, preds)
                            return {
                                "accuracy": acc,
                                "f1": f1,
                                "precision": precision,
                                "recall": recall,
                            }

                        training_args = TrainingArguments(
                            output_dir="./niyet_bert_results",
                            eval_strategy="epoch",
                            save_strategy="epoch",
                            learning_rate=learning_rate,
                            per_device_train_batch_size=batch_size,
                            per_device_eval_batch_size=batch_size,
                            num_train_epochs=epochs,
                            weight_decay=0.01,
                            load_best_model_at_end=True,
                            metric_for_best_model="f1",
                            logging_steps=10,
                        )

                        trainer = Trainer(
                            model=model,
                            args=training_args,
                            train_dataset=tokenized_datasets["train"],
                            eval_dataset=tokenized_datasets["test"],
                            compute_metrics=compute_metrics,
                        )
                        trainer.train()

                        eval_results = trainer.evaluate(tokenized_datasets["test"])
                        test_preds_output = trainer.predict(tokenized_datasets["test"])
                        y_pred = np.argmax(test_preds_output.predictions, axis=1)
                        y_true = test_preds_output.label_ids
                        cm = confusion_matrix(y_true, y_pred)

                        save_dir = "./niyet_bert_model"
                        trainer.save_model(save_dir)
                        tokenizer.save_pretrained(save_dir)
                        np.save(
                            "./niyet_bert_model/classes.npy", label_encoder.classes_
                        )

                        st.session_state.model = model
                        st.session_state.tokenizer = tokenizer
                        st.session_state.label_encoder = label_encoder
                        st.session_state.eval_results = eval_results
                        st.session_state.cm = cm
                        st.session_state.classes = label_encoder.classes_
                        st.success("Model başarıyla eğitildi ve kaydedildi.")

        else:
            def find_local_model_folders(base_path="."):
                model_dirs = []
                try:
                    for root, dirs, files in os.walk(base_path):
                        dirs[:] = [d for d in dirs if not d.startswith((".", "__"))]
                        if "config.json" in files or "classes.npy" in files or "model.safetensors" in files:
                            rel_p = os.path.relpath(root, base_path).replace("\\", "/")
                            model_dirs.append(f"./{rel_p}" if rel_p != "." else "./")
                except Exception:
                    pass
                if not model_dirs:
                    model_dirs = ["./niyet_bert_model"]
                return sorted(list(set(model_dirs)))

            bulunan_modeller = find_local_model_folders(".")

            if "selected_model_path" not in st.session_state:
                st.session_state.selected_model_path = "./niyet_bert_model"

            st.markdown("##### Model Klasörü Seçimi")

            # Dropdown: Otomatik bulunan modeller
            secenekler = bulunan_modeller + ["✍️ Başka bir yol girmek istiyorum"]
            suanki_idx = 0
            if st.session_state.selected_model_path in secenekler:
                suanki_idx = secenekler.index(st.session_state.selected_model_path)

            secilen_dropdown = st.selectbox(
                "🔍 Tespit Edilen Model Klasörleri:",
                secenekler,
                index=suanki_idx,
                help="Proje dizininde otomatik bulunan BERT model klasörleri. Listede yoksa alttaki kutuya tam yolu yazın.",
            )

            # Dropdown'dan seçim yapıldıysa text input'u güncelle
            if secilen_dropdown != "✍️ Başka bir yol girmek istiyorum":
                st.session_state.selected_model_path = secilen_dropdown

            # Manuel yol girişi
            model_path = st.text_input(
                "✏️ Model Klasör Yolu (listedeki seçimi değiştirmek veya farklı yol girmek için):",
                value=st.session_state.selected_model_path,
                key="model_path_input",
                placeholder="Örn: ./niyet_bert_model veya C:/modeller/bert_niyet",
            )
            st.session_state.selected_model_path = model_path

            # Yol bilgisi/uyarı
            if model_path and os.path.exists(model_path):
                st.caption(f"✅ Klasör mevcut: `{os.path.abspath(model_path)}`")
            elif model_path:
                st.caption(f"⚠️ Klasör bulunamadı: `{model_path}`")

            if st.button("🚀 Modeli Yükle", type="primary", use_container_width=True):
                if os.path.exists(model_path):
                    with st.spinner("Model yükleniyor..."):
                        try:
                            tokenizer = AutoTokenizer.from_pretrained(model_path)
                            device = "cuda" if torch.cuda.is_available() else "cpu"
                            model = AutoModelForSequenceClassification.from_pretrained(
                                model_path
                            ).to(device)
                            classes_file = os.path.join(model_path, "classes.npy")
                            if os.path.exists(classes_file):
                                classes = np.load(classes_file, allow_pickle=True)
                            elif hasattr(model.config, "id2label") and model.config.id2label:
                                classes = [model.config.id2label[i] for i in sorted(model.config.id2label.keys())]
                            else:
                                classes = np.array([f"Sinif_{i}" for i in range(model.config.num_labels)])

                            # Gerçek metrikleri trainer_state.json'dan oku
                            def metrikleri_yukle(model_klasoru):
                                aralinacak_yerler = [
                                    model_klasoru,
                                    os.path.join(os.path.dirname(model_klasoru), "niyet_bert_results"),
                                    "./niyet_bert_results",
                                ]
                                for yer in aralinacak_yerler:
                                    if not os.path.exists(yer):
                                        continue

                                    # Doğrudan trainer_state.json var mı?
                                    state_file = os.path.join(yer, "trainer_state.json")
                                    if os.path.exists(state_file):
                                        with open(state_file, "r", encoding="utf-8") as f:
                                            state = json.load(f)
                                        log_history = state.get("log_history", [])
                                        eval_logs = [e for e in log_history if "eval_accuracy" in e]
                                        if eval_logs:
                                            return max(eval_logs, key=lambda x: x.get("eval_f1", 0))

                                    # Alt klasörlerdeki checkpoint'leri SAYISAL olarak sırala (en büyük step = son epoch)
                                    checkpoints = []
                                    for item in os.listdir(yer):
                                        item_path = os.path.join(yer, item)
                                        if os.path.isdir(item_path) and item.startswith("checkpoint-"):
                                            try:
                                                step_no = int(item.split("-")[-1])
                                                checkpoints.append((step_no, item_path))
                                            except ValueError:
                                                pass

                                    # En büyük step numarasından başlayarak oku
                                    for _, cp_path in sorted(checkpoints, reverse=True):
                                        cp_state = os.path.join(cp_path, "trainer_state.json")
                                        if os.path.exists(cp_state):
                                            with open(cp_state, "r", encoding="utf-8") as f:
                                                state = json.load(f)
                                            log_history = state.get("log_history", [])
                                            eval_logs = [e for e in log_history if "eval_accuracy" in e]
                                            if eval_logs:
                                                # En yüksek F1 skorlu logu döndür
                                                return max(eval_logs, key=lambda x: x.get("eval_f1", 0))
                                return None

                            gercek_metrikler = metrikleri_yukle(model_path)

                            label_encoder = LabelEncoder()
                            label_encoder.classes_ = classes
                            st.session_state.model = model
                            st.session_state.tokenizer = tokenizer
                            st.session_state.label_encoder = label_encoder
                            st.session_state.classes = label_encoder.classes_
                            st.session_state.cm = None  # Yüklemede cm yok

                            if gercek_metrikler:
                                st.session_state.eval_results = gercek_metrikler
                                st.session_state.metrics_source = "trainer_state"
                            else:
                                st.session_state.eval_results = None
                                st.session_state.metrics_source = "none"

                            st.success(f"✅ Model başarıyla yüklendi! ({len(classes)} sınıf · Cihaz: {device.upper()})")
                            if gercek_metrikler:
                                acc = gercek_metrikler.get("eval_accuracy", 0)
                                f1 = gercek_metrikler.get("eval_f1", 0)
                                st.info(f"📊 Performans sonuçları yüklendi — Accuracy: **{acc:.4f}** · F1: **{f1:.4f}**")
                            else:
                                st.warning("⚠️ Bu model için kayıtlı performans sonucu bulunamadı. '02 · Performans' sekmesinde değerlendirme yapabilirsiniz.")
                        except Exception as err:
                            st.error(f"Model yüklenirken hata oluştu: {err}")
                else:
                    st.error(f"❌ Belirtilen klasör bulunamadı: `{model_path}`")


with tab2:
    with st.container():
        st.header("Model Performans Değerlendirmesi")
        if st.session_state.eval_results is not None:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric(
                "Accuracy",
                f"{st.session_state.eval_results.get('eval_accuracy', 0):.4f}",
            )
            col2.metric(
                "F1 Score", f"{st.session_state.eval_results.get('eval_f1', 0):.4f}"
            )
            col3.metric(
                "Precision",
                f"{st.session_state.eval_results.get('eval_precision', 0):.4f}",
            )
            col4.metric(
                "Recall", f"{st.session_state.eval_results.get('eval_recall', 0):.4f}"
            )

            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            if st.session_state.cm is not None:
                bg = "#F5F9FD"
                grid = "#4B5A6B"
                cm = st.session_state.cm
                fig_cm, ax_cm = plt.subplots(figsize=(7, 4.5))
                fig_cm.patch.set_facecolor(bg)
                ax_cm.set_facecolor(bg)
                sns.heatmap(
                    cm,
                    annot=True,
                    fmt="d",
                    cmap="Blues",
                    xticklabels=st.session_state.classes,
                    yticklabels=st.session_state.classes,
                    ax=ax_cm,
                    cbar=False,
                    annot_kws={"fontweight": "bold", "fontsize": 10},
                    linewidths=1,
                    linecolor=bg,
                )
                # Koyu hücrelerde beyaz, açık hücrelerde koyu lacivert metin (okunabilirlik için)
                esik = cm.max() * 0.6 if cm.max() > 0 else 0
                for text_obj, deger in zip(ax_cm.texts, cm.flatten()):
                    text_obj.set_color("#FFFFFF" if deger > esik else "#10151C")
                ax_cm.set_title(
                    "Confusion Matrix",
                    fontsize=12,
                    fontweight="bold",
                    color=grid,
                    pad=15,
                )
                ax_cm.tick_params(colors=grid, labelsize=9)
                plt.setp(ax_cm.get_xticklabels(), rotation=30, ha="right")
                for spine in ax_cm.spines.values():
                    spine.set_visible(False)
                st.pyplot(fig_cm)
        else:
            st.info(
                "Görüntülenecek performans verisi yok. Önce “01 · Veri & Model” sekmesinden bir model yükleyin."
            )


with tab3:
    with st.container():
        st.header("Manuel Niyet Tahmini & Detaylı Analiz")
        if st.session_state.model is not None:
            user_input = st.text_area(
                "Test etmek istediğiniz cümleyi girin:",
                "İntegral formüllerini özetler misin?",
            )
            if st.button("Tahmin Et", type="primary"):
                model = st.session_state.model
                tokenizer = st.session_state.tokenizer
                label_encoder = st.session_state.label_encoder

                inputs = tokenizer(
                    user_input,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=128,
                )
                inputs = {k: v.to(model.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits
                    probabilities = torch.softmax(logits, dim=-1).cpu().tolist()[0]
                    pred_id = torch.argmax(logits, dim=-1).item()

                predicted_label = label_encoder.inverse_transform([pred_id])[0]
                st.markdown(
                    f"<br><span class='metric-badge'>TAHMİN — {predicted_label}</span><br><br>",
                    unsafe_allow_html=True,
                )

                st.markdown("##### Sınıf Olasılık Dağılımı")
                prob_dict = {
                    cls_name: prob
                    for cls_name, prob in zip(label_encoder.classes_, probabilities)
                }
                sirali_probs = sorted(
                    prob_dict.items(), key=lambda x: x[1], reverse=True
                )

                rows_html = ""
                for i, (sinif_adi, olasilik) in enumerate(sirali_probs):
                    fill_class = "prob-fill top" if i == 0 else "prob-fill"
                    rows_html += f"""
                    <div class='prob-row'>
                        <div class='prob-label' title='{sinif_adi}'>{sinif_adi}</div>
                        <div class='prob-track'><div class='{fill_class}' style='width:{olasilik * 100:.1f}%'></div></div>
                        <div class='prob-value'>%{olasilik * 100:.1f}</div>
                    </div>
                    """
                st.markdown(rows_html, unsafe_allow_html=True)
        else:
            st.warning("Test yapmak için önce model yükleyin.")


with tab4:
    with st.container():
        st.header("Excel Tabanlı Parametrik DPO Veri Toplama")
        st.markdown(
            "<p style='color:var(--text-muted)'>Parametreleri belirleyin, sistem akıllı çeldiriciler üretsin. Onaylayıp <b>“Kaydet ve Sıradakine Geç”</b> dediğinizde çift, veri havuzuna eklenir.</p>",
            unsafe_allow_html=True,
        )

        col_api1, col_api2 = st.columns([2, 1])
        with col_api1:
            gemini_api_key = st.text_input(
                "Google Gemini API Key",
                type="password",
                placeholder="AIzaSy...",
                key="dpo_key",
            )
        with col_api2:
            gemini_model_secimi = st.selectbox(
                "Kullanılacak LLM Modeli",
                [
                    "gemini-2.5-flash",
                    "gemini-2.5-flash-lite",
                    "gemini-2.5-pro",
                    "gemini-3.5-flash",
                    "gemini-3.5-flash-lite",
                    "gemini-3.7-flash",
                    "gemini-3.1-pro-preview",
                ],
                index=0,
            )

        st.markdown("---")
        st.markdown("##### Üretim Parametreleri")
        p_col1, p_col2, p_col3 = st.columns(3)

        with p_col1:
            kategori_opsiyonlari = ["Rastgele (Otomatik)"]
            if st.session_state.classes is not None:
                kategori_opsiyonlari.extend(list(st.session_state.classes))
            secilen_hedef_kategori = st.selectbox(
                "Hedef Niyet / Kategori", kategori_opsiyonlari, index=0
            )

        with p_col2:
            secilen_zorluk = st.selectbox(
                "Zorluk / Çeldirici Seviyesi",
                ["Aşırı Zorlayıcı & Çeldiricili", "Standart / Doğal"],
                index=0,
            )

        with p_col3:
            secilen_uzunluk = st.selectbox(
                "Soru Uzunluğu",
                ["Kısa (Tek Cümle)", "Orta (Standart)", "Uzun (Detaylı/Bağlamlı)"],
                index=1,
            )

        excel_path = "Ders_konu_bilissel-seviye_öğrenme-çıktısı.xlsx"

        if st.session_state.model is not None:

            def modelle_tahmin_et(metin):
                if not metin or not metin.strip():
                    return None, []
                model = st.session_state.model
                tokenizer = st.session_state.tokenizer
                label_encoder = st.session_state.label_encoder

                inputs = tokenizer(
                    metin,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=128,
                )
                inputs = {k: v.to(model.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits
                    probs = torch.softmax(logits, dim=-1).cpu().tolist()[0]
                    pred_id = torch.argmax(logits, dim=-1).item()

                tahmin = label_encoder.inverse_transform([pred_id])[0]
                prob_dict = {
                    cls_name: prob
                    for cls_name, prob in zip(label_encoder.classes_, probs)
                }
                sirali_probs = sorted(
                    prob_dict.items(), key=lambda x: x[1], reverse=True
                )
                return tahmin, sirali_probs

            def otomatik_soru_uret(kapsam_disi_mod="tamamen_farkli"):
                """
                kapsam_disi_mod:
                  'tamamen_farkli' → günlük hayat konusu (okulla hiç ilgisi yok)
                  'excel'          → Excel'deki ders konusunu kullan ama kapsam dışı bir talep yaz
                """
                if not gemini_api_key:
                    st.error("Otomatik soru üretimi için lütfen yukarıdaki alana Gemini API anahtarınızı girin.")
                    return False
                try:
                    df_excel = pd.read_excel(excel_path)
                    if "Konu Adı" not in df_excel.columns:
                        st.error("Okunan dosyada 'Konu Adı' sütunu bulunamadı.")
                        return False

                    konular = df_excel["Konu Adı"].dropna().astype(str).tolist()
                    secilen_konu = random.choice(konular)
                    st.session_state.secilen_excel_konusu = secilen_konu

                    if secilen_hedef_kategori == "Rastgele (Otomatik)":
                        hedef_niyet = random.choice(list(st.session_state.classes))
                    else:
                        hedef_niyet = secilen_hedef_kategori

                    # Her çağrıda rastgele çeşitlilik: üslup (normal kategori üretiminde kullanılır)
                    secilen_uslup = random.choice(URETIM_USLUPLARI)

                    if "Kısa" in secilen_uzunluk:
                        uzunluk_talimati = "Cümle tek parça, kısa ve öz olsun (3-10 kelime arası)."
                    elif "Uzun" in secilen_uzunluk:
                        uzunluk_talimati = "Cümle detaylı, bağlam içeren uzun bir cümle olsun."
                    else:
                        uzunluk_talimati = "Cümle standart orta uzunlukta olsun."

                    client = genai.Client(api_key=gemini_api_key)

                    # Kapsam_Dışı tespiti
                    _siniflar = list(st.session_state.classes) if st.session_state.classes is not None else []
                    hedef_kapsam_disi = (
                        "kapsam" in hedef_niyet.lower()
                        or "disi" in hedef_niyet.lower()
                        or "dışı" in hedef_niyet.lower()
                    )

                    if hedef_kapsam_disi and kapsam_disi_mod == "excel":
                        # Mod 1: Excel'deki ders konusunu kullan ama kapsam dışı talep yaz
                        # Örn: Matematik konusunda ama "bunu arkadaşıma nasıl anlatabilirim?"
                        secilen_talep = random.choice(KAPSAM_DISI_TALEPLERI)
                        prompt_to_gemini = (
                            f"Bir yapay zeka eğitim asistanı sistemini test ediyorsun. "
                            f"Sistem yalnızca şu kategorilerdeki istekleri işliyor: {', '.join(_siniflar[:6])} vb.\n"
                            f"GÖREV: '{secilen_konu}' konusundan bahseden ama sisteme uygun bir DERS TALEBİ (soru sorma, "
                            f"alıştırma isteme, açıklama isteme vb.) olmayan, bunun yerine '{secilen_talep}' "
                            f"şeklinde bir mesaj yaz. Bu mesaj sisteme kapsam dışı görünmeli. Diğer sınıfların kelimeleri "
                            f"(örn. 'özetin çok işime yaradı, teşekkürler' gibi) çeldirici/yan unsur olarak geçebilir.\n"
                            f"ÜSLUP: {secilen_uslup}\n"
                            f"UZUNLUK: {uzunluk_talimati}\n"
                            f"KURAL: Sadece öğrencinin kuracağı TEK BİR doğal cümle ver. Kategori adını yazma."
                        )
                        st.session_state.secilen_excel_konusu = f"{secilen_konu} → [Excel Konusu, Kapsam Dışı]"

                    elif hedef_kapsam_disi and kapsam_disi_mod == "tamamen_farkli":
                        # Mod 2: Okulla hiç ilgisi olmayan günlük hayat sorusu
                        kapsam_disi_temalar = [
                            "günlük alışveriş (market, sebze, meyve fiyatları)",
                            "hava durumu ve mevsimler",
                            "yemek tarifi veya restoranlar",
                            "spor maçları ve sporcular",
                            "film, dizi veya müzik",
                            "seyahat ve tatil planları",
                            "araç, taşıt veya trafik",
                            "sağlık, ilaç veya doktor",
                            "moda, kıyafet ve alışveriş",
                            "evcil hayvanlar ve doğa",
                            "politik haberler veya ekonomi",
                            "sosyal medya ve internet",
                        ]
                        kapsam_disi_tema = random.choice(kapsam_disi_temalar)
                        prompt_to_gemini = (
                            f"Bir yapay zeka eğitim asistanı sistemini test ediyorsun. "
                            f"Bu sistem yalnızca ders/eğitim sorularını yanıtlamak için tasarlanmış.\n"
                            f"GÖREV: Eğitimle, okulla, derslerle veya akademik konularla HİÇBİR İLGİSİ OLMAYAN, "
                            f"tamamen alakasız ve konu dışı bir mesaj yaz. Tema: {kapsam_disi_tema}\n"
                            f"ÜSLUP: {secilen_uslup} Kısa ve doğal sohbet cümleleri de olabilir.\n"
                            f"UZUNLUK: {uzunluk_talimati}\n"
                            f"KURAL: Mesajda ders, ödev, konu, formül, öğretmen, okul gibi eğitimle ilgili hiçbir "
                            f"kelime KULLANMA. Sadece sıradan bir kullanıcının kuracağı TEK BİR cümle ver."
                        )
                        st.session_state.secilen_excel_konusu = f"[Kapsam Dışı - {kapsam_disi_tema}]"

                    else:
                        # Normal niyet kategorisi
                        hedef_tanim_metni = _hedef_tanimini_formatla(hedef_niyet)
                        yasakli_ifadeler_metni = ", ".join(f'"{ifade}"' for ifade in YASAKLI_IFADELER)
                        uretilmeyecek_tur_metni = (
                            "ÜRETME: Öğrencinin konu hakkında açıklama/bilgi istediği cümleler "
                            "(\"bana X'i anlat\", \"...nasıl oldu merak ediyorum\") ÜRETME — bu tür istekler "
                            "bu görevin kapsamında değil."
                        )

                        if "Aşırı Zorlayıcı" in secilen_zorluk:
                            # Hedef dışındaki sınıflardan rastgele bir çeldirici sınıf seç
                            diger_siniflar = [s for s in _siniflar if s != hedef_niyet]
                            celdirici_sinif = random.choice(diger_siniflar) if diger_siniflar else hedef_niyet
                            celdirici_tanim_metni = _celdirici_sinifi_formatla(celdirici_sinif)
                            celdirici_teknik = random.choice(CELDIRICI_TEKNIKLERI).format(
                                celdirici=celdirici_sinif, hedef=hedef_niyet
                            )
                            gorev_metni = (
                                f"GÖREV: Cümlenin ASIL isteği {hedef_niyet} olmalı. Yanıtta {celdirici_sinif} sınıfının "
                                f"kelimeleri yalnızca YAN UNSUR olarak geçebilir (olumsuzlanmış, reddedilmiş, geçmişte "
                                f"kalmış veya ertelenmiş bir istek olarak). İki isteği EŞİT AĞIRLIKTA birleştirme.\n"
                                f"ÇELDİRİCİ TEKNİK: {celdirici_teknik}"
                            )
                            prompt_to_gemini = (
                                f"Sen liseli bir öğrencisin. Yapay zeka öğretmenine '{secilen_konu}' konusuyla ilgili mesaj yaz.\n\n"
                                f"{hedef_tanim_metni}\n\n"
                                f"{celdirici_tanim_metni}\n\n"
                                f"{gorev_metni}\n"
                                f"ÜSLUP: {secilen_uslup}\n"
                                f"UZUNLUK: {uzunluk_talimati}\n"
                                f"YASAKLI İFADELER: Şu kelime/ifadeleri KULLANMA: {yasakli_ifadeler_metni}.\n"
                                f"{uretilmeyecek_tur_metni}\n"
                                f"KURAL: Asla kategori adını metne yansıtma. Sadece öğrencinin kuracağı TEK BİR cümle ver."
                            )
                        else:
                            prompt_to_gemini = (
                                f"Sen liseli bir öğrencisin. Yapay zeka öğretmenine '{secilen_konu}' konusuyla ilgili mesaj yaz.\n\n"
                                f"{hedef_tanim_metni}\n\n"
                                f"GÖREV: '{hedef_niyet}' kategorisine uygun doğal bir talep ilet.\n"
                                f"ÜSLUP: {secilen_uslup}\n"
                                f"UZUNLUK: {uzunluk_talimati}\n"
                                f"YASAKLI İFADELER: Şu kelime/ifadeleri KULLANMA: {yasakli_ifadeler_metni}.\n"
                                f"{uretilmeyecek_tur_metni}\n"
                                f"KURAL: Asla kategori adını metne yansıtma. Sadece öğrencinin kuracağı TEK BİR cümle ver."
                            )

                    # Gemini çağrısı — yasaklı ifade geçerse bir kez yeniden üret
                    generated_text = ""
                    for _deneme in range(2):
                        response = client.models.generate_content(
                            model=gemini_model_secimi, contents=prompt_to_gemini
                        )
                        generated_text = _gemini_ciktisini_temizle(response.text)
                        if not _yasakli_ifade_var_mi(generated_text):
                            break

                    st.session_state.generated_test_text = generated_text

                    tahmin, sirali_probs = modelle_tahmin_et(generated_text)
                    st.session_state.bert_prediction = tahmin
                    st.session_state.bert_probabilities = sirali_probs
                    return True

                except Exception as e:
                    hata_mesaji = str(e)
                    if "429" in hata_mesaji or "RESOURCE_EXHAUSTED" in hata_mesaji:
                        st.error(
                            "API hız sınırına ulaşıldı. Lütfen farklı bir model seçin veya 1 dakika bekleyin."
                        )
                    else:
                        st.error(f"Hata oluştu: {hata_mesaji}")
                    return False

            if "generated_test_text" not in st.session_state:
                st.session_state.generated_test_text = ""

            if st.session_state.get("secilen_excel_konusu"):
                st.markdown(
                    f"<span class='status-badge'>KONU · {st.session_state.get('secilen_excel_konusu', '')}</span>",
                    unsafe_allow_html=True,
                )
                st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

            guncel_soru = st.text_area(
                "Değerlendirilecek Soru / Cümle (Gemini'nin ürettiğini düzenleyebilir veya doğrudan kendi cümlenizi yazabilirsiniz):",
                value=st.session_state.generated_test_text,
                height=90,
                placeholder="Örn: Yapay zeka öğretmenim, bu formülün nereden geldiğini açıklar mısınız?",
            )

            # Kapsam_Dışı tespiti (buton düzeni için)
            _hedef = secilen_hedef_kategori if secilen_hedef_kategori != "Rastgele (Otomatik)" else ""
            _kapsam_secili = (
                "kapsam" in _hedef.lower()
                or "disi" in _hedef.lower()
                or "dışı" in _hedef.lower()
            )

            if _kapsam_secili and gemini_api_key:
                # Kapsam_Dışı seçili → tam satır tahmin butonu + altında iki üretim modu
                if st.button("🔍 Yazılan Cümleyi Modele Tahmin Ettir", type="primary", use_container_width=True):
                    if guncel_soru.strip():
                        st.session_state.generated_test_text = guncel_soru
                        tahmin, sirali_probs = modelle_tahmin_et(guncel_soru)
                        st.session_state.bert_prediction = tahmin
                        st.session_state.bert_probabilities = sirali_probs
                        st.rerun()
                    else:
                        st.warning("Lütfen tahmin için bir metin girin.")

                col_kd1, col_kd2 = st.columns(2)
                with col_kd1:
                    if st.button("📚 Gemini ile Üret — Excel Konusu (Zor/Çeldirici)",
                                 use_container_width=True,
                                 help="Excel'deki bir ders konusunu kullanır ama kapsam dışı bir talep yazar. Model için zor örnekler."):
                        st.session_state.son_kapsam_disi_mod = "excel"
                        with st.spinner("Soru hazırlanıyor..."):
                            if otomatik_soru_uret(kapsam_disi_mod="excel"):
                                st.rerun()
                with col_kd2:
                    if st.button("🌀 Gemini ile Üret — Günlük Hayat Konusu",
                                 use_container_width=True,
                                 help="Okulla hiç ilgisi olmayan günlük hayat sorusu üretir."):
                        st.session_state.son_kapsam_disi_mod = "tamamen_farkli"
                        with st.spinner("Soru hazırlanıyor..."):
                            if otomatik_soru_uret(kapsam_disi_mod="tamamen_farkli"):
                                st.rerun()
            else:
                # Normal kategori → iki buton yan yana (eski düzen)
                col_act1, col_act2 = st.columns(2)
                with col_act1:
                    if st.button("🔍 Yazılan Cümleyi Modele Tahmin Ettir", type="primary", use_container_width=True):
                        if guncel_soru.strip():
                            st.session_state.generated_test_text = guncel_soru
                            tahmin, sirali_probs = modelle_tahmin_et(guncel_soru)
                            st.session_state.bert_prediction = tahmin
                            st.session_state.bert_probabilities = sirali_probs
                            st.rerun()
                        else:
                            st.warning("Lütfen tahmin için bir metin girin.")
                with col_act2:
                    if st.button("🎲 Gemini ile Yeni Soru Üret", use_container_width=True):
                        with st.spinner("Yeni soru hazırlanıyor..."):
                            if otomatik_soru_uret():
                                st.rerun()

            # Eğer kullanıcı metni değiştirdiyse ancak butona basmadıysa da güncel tutalım
            if guncel_soru != st.session_state.generated_test_text:
                st.session_state.generated_test_text = guncel_soru

            # Olasılık Dağılımı ve Tahmin Kartı
            if st.session_state.get("bert_prediction"):
                st.markdown("---")
                col_tahmin, col_dagilim = st.columns([1, 2])
                with col_tahmin:
                    st.markdown("##### Model Tahmini")
                    st.markdown(
                        f"<span class='tag-rejected' style='font-size:14px; padding:8px 14px;'>{st.session_state.bert_prediction} · REJECTED</span>",
                        unsafe_allow_html=True,
                    )
                    st.caption("Model bu tahmini yaptı. Eğer bu tahmin yanlışsa DPO çiftinde 'REJECTED' olarak kaydedilecektir.")

                with col_dagilim:
                    st.markdown("##### Model Olasılık Dağılımı")
                    if "bert_probabilities" in st.session_state and st.session_state.bert_probabilities:
                        rows_html = ""
                        for i, (sinif_adi, olasilik) in enumerate(
                            st.session_state.bert_probabilities
                        ):
                            fill_class = "prob-fill top" if i == 0 else "prob-fill"
                            rows_html += f"""
                            <div class='prob-row'>
                                <div class='prob-label' title='{sinif_adi}'>{sinif_adi}</div>
                                <div class='prob-track'><div class='{fill_class}' style='width:{olasilik * 100:.1f}%'></div></div>
                                <div class='prob-value'>%{olasilik * 100:.1f}</div>
                            </div>
                            """
                        st.markdown(rows_html, unsafe_allow_html=True)

                st.markdown("---")

                # CHOSEN seçimi: Eğer yukarıda belirli bir hedef kategori seçilmişse onu varsayılan yap
                default_idx = 0
                if secilen_hedef_kategori != "Rastgele (Otomatik)" and secilen_hedef_kategori in st.session_state.classes:
                    default_idx = list(st.session_state.classes).index(secilen_hedef_kategori)
                elif st.session_state.bert_prediction in st.session_state.classes:
                    default_idx = list(st.session_state.classes).index(
                        st.session_state.bert_prediction
                    )

                gercek_dogru_kategori = st.selectbox(
                    "Nihai karar — gerçekte olması gereken CHOSEN kategori:",
                    st.session_state.classes,
                    index=default_idx,
                )

                st.markdown("<br>", unsafe_allow_html=True)
                col_btn1, col_btn2 = st.columns(2)

                if col_btn1.button(
                    "💾 Kaydet ve Sıradakine Geç", type="primary", use_container_width=True
                ):
                    # En güncel metni ve tahmini al
                    aktif_metin = guncel_soru.strip()
                    if not aktif_metin:
                        st.error("Metin kutusu boş olamaz.")
                    else:
                        aktif_tahmin, _ = modelle_tahmin_et(aktif_metin)
                        if aktif_tahmin == gercek_dogru_kategori:
                            st.error(
                                f"Model tahmini ({aktif_tahmin}) ile seçtiğiniz CHOSEN kategori ({gercek_dogru_kategori}) aynı. DPO çifti için modelin hatalı tahmini (REJECTED) ile gerçek doğru (CHOSEN) farklı olmalıdır."
                            )
                        else:
                            dpo_pair = {
                                "prompt": aktif_metin,
                                "chosen": gercek_dogru_kategori,
                                "rejected": aktif_tahmin,
                                "konu": st.session_state.get("secilen_excel_konusu", "Manuel Giriş"),
                            }
                            st.session_state.dpo_dataset_buffer.append(dpo_pair)

                            # Diske anında kaydet (append mod — üzerine ekler)
                            try:
                                with open(DPO_OUTPUT_FILE, "a", encoding="utf-8") as f_out:
                                    f_out.write(json.dumps(dpo_pair, ensure_ascii=False) + "\n")
                                disk_kayit_ok = True
                            except Exception as disk_err:
                                disk_kayit_ok = False
                                st.warning(f"⚠️ Diske yazılamadı: {disk_err}")

                            with st.status(
                                "İşlem kaydediliyor...",
                                expanded=True,
                            ) as status:
                                if disk_kayit_ok:
                                    st.write(f"✅ Diske kaydedildi → `{DPO_OUTPUT_FILE}`")
                                else:
                                    st.write("💾 Tampon belleğe eklendi (disk yazımı başarısız).")
                                if gemini_api_key:
                                    otomatik_soru_uret(
                                        kapsam_disi_mod=st.session_state.get("son_kapsam_disi_mod", "tamamen_farkli")
                                    )
                                else:
                                    st.session_state.generated_test_text = ""
                                    st.session_state.bert_prediction = None
                                    st.session_state.bert_probabilities = []
                                status.update(
                                    label="Hazır.", state="complete", expanded=False
                                )
                            st.rerun()

                if col_btn2.button(
                    "⏭️ Soruyu At ve Yenisini Üret", use_container_width=True
                ):
                    with st.status(
                        "Soru atlanıyor, yenisi hazırlanıyor...", expanded=True
                    ) as status:
                        if gemini_api_key:
                            otomatik_soru_uret(
                                kapsam_disi_mod=st.session_state.get("son_kapsam_disi_mod", "tamamen_farkli")
                            )
                        else:
                            st.session_state.generated_test_text = ""
                            st.session_state.bert_prediction = None
                            st.session_state.bert_probabilities = []
                        status.update(label="Hazır.", state="complete", expanded=False)
                    st.rerun()

            # --- DPO Veri Havuzu Yönetimi (her zaman göster, dosya varsa) ---
            st.markdown("---")
            st.subheader("📦 DPO Veri Havuzu Yönetimi")

            # Disk dosyasından toplam kayıt sayısını oku
            disk_kayit_sayisi = 0
            if os.path.exists(DPO_OUTPUT_FILE):
                try:
                    with open(DPO_OUTPUT_FILE, "r", encoding="utf-8") as f_read:
                        disk_kayit_sayisi = sum(1 for line in f_read if line.strip())
                except Exception:
                    disk_kayit_sayisi = 0

            oturum_kayit_sayisi = len(st.session_state.dpo_dataset_buffer)

            col_stat1, col_stat2, col_stat3 = st.columns(3)
            col_stat1.metric("Bu Oturumda Eklenen", f"{oturum_kayit_sayisi} çift")
            col_stat2.metric("Dosyada Toplam Kayıt", f"{disk_kayit_sayisi} çift")
            col_stat3.metric(
                "Kayıt Dosyası",
                os.path.basename(DPO_OUTPUT_FILE),
                help=DPO_OUTPUT_FILE,
            )

            if disk_kayit_sayisi > 0 or oturum_kayit_sayisi > 0:
                st.caption(f"📁 Dosya konumu: `{DPO_OUTPUT_FILE}`")
                st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

                # Disk dosyasını oku (indirme için)
                try:
                    if os.path.exists(DPO_OUTPUT_FILE):
                        with open(DPO_OUTPUT_FILE, "r", encoding="utf-8") as f_dl:
                            disk_icerik = f_dl.read()
                    else:
                        disk_icerik = "\n".join(
                            json.dumps(item, ensure_ascii=False)
                            for item in st.session_state.dpo_dataset_buffer
                        )
                except Exception:
                    disk_icerik = ""

                col_down1, col_down2, col_down3 = st.columns(3)
                with col_down1:
                    st.download_button(
                        label=f"⬇️ Tüm DPO Verisini İndir ({disk_kayit_sayisi} kayıt)",
                        data=disk_icerik,
                        file_name="dpo_dataset.jsonl",
                        mime="text/plain",
                        type="primary",
                        use_container_width=True,
                        disabled=(len(disk_icerik.strip()) == 0),
                    )
                with col_down2:
                    if st.button("🗑️ Hafızayı Temizle (Oturum)", use_container_width=True):
                        st.session_state.dpo_dataset_buffer = []
                        st.rerun()
                with col_down3:
                    if st.button("⚠️ Disk Dosyasını Sıfırla", use_container_width=True):
                        try:
                            if os.path.exists(DPO_OUTPUT_FILE):
                                os.remove(DPO_OUTPUT_FILE)
                            st.session_state.dpo_dataset_buffer = []
                            st.success("Disk dosyası silindi, hafıza temizlendi.")
                            st.rerun()
                        except Exception as del_err:
                            st.error(f"Dosya silinemedi: {del_err}")
            else:
                st.info("Henüz kaydedilmiş DPO verisi yok. 'Kaydet ve Sıradakine Geç' ile veri eklemeye başlayın.")
        else:
            st.warning(
                "Lütfen önce model yükleyin ve yukarıdaki alana Gemini API anahtarınızı girin."
            )
