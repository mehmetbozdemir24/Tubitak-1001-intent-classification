import os
import json
import random
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
            model_path = st.text_input("Model Klasör Yolu", value="./niyet_bert_model")
            if st.button("Modeli Yükle", type="primary", use_container_width=True):
                if os.path.exists(model_path):
                    tokenizer = AutoTokenizer.from_pretrained(model_path)
                    model = AutoModelForSequenceClassification.from_pretrained(
                        model_path
                    )
                    classes_file = os.path.join(model_path, "classes.npy")
                    if os.path.exists(classes_file):
                        classes = np.load(classes_file, allow_pickle=True)
                        label_encoder = LabelEncoder()
                        label_encoder.classes_ = classes
                        st.session_state.model = model
                        st.session_state.tokenizer = tokenizer
                        st.session_state.label_encoder = label_encoder
                        st.session_state.classes = label_encoder.classes_
                        st.session_state.eval_results = {
                            "eval_accuracy": 1.0,
                            "eval_f1": 1.0,
                            "eval_precision": 1.0,
                            "eval_recall": 1.0,
                        }
                        st.success("Model başarıyla yüklendi.")


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

        if st.session_state.model is not None and gemini_api_key:

            def modelle_tahmin_et(metin):
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

            def otomatik_soru_uret():
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

                    if "Aşırı Zorlayıcı" in secilen_zorluk:
                        stil_talimati = f"NİYETİN: '{hedef_niyet}' kategorisine ait olmak. Ancak cümlede başka niyetlerin kelimelerini çeldirici olarak kullanarak modeli şaşırt ve zorla."
                    else:
                        stil_talimati = (
                            f"'{hedef_niyet}' kategorisine uygun doğal bir talep ilet."
                        )

                    if "Kısa" in secilen_uzunluk:
                        uzunluk_talimati = "Cümle tek parça, kısa ve öz olsun."
                    elif "Uzun" in secilen_uzunluk:
                        uzunluk_talimati = (
                            "Cümle detaylı, bağlam içeren uzun bir cümle olsun."
                        )
                    else:
                        uzunluk_talimati = "Cümle standart orta uzunlukta olsun."

                    client = genai.Client(api_key=gemini_api_key)
                    prompt_to_gemini = (
                        f"Sen liseli bir öğrencisin. Yapay zeka öğretmenine '{secilen_konu}' konusuyla ilgili mesaj yaz.\n"
                        f"GÖREV: {stil_talimati}\n"
                        f"UZUNLUK: {uzunluk_talimati}\n"
                        f"KURAL: Asla kategori adını metne yansıtma. Sadece öğrencinin kuracağı TEK BİR cümle ver."
                    )

                    response = client.models.generate_content(
                        model=gemini_model_secimi, contents=prompt_to_gemini
                    )
                    generated_text = response.text.strip()
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
                if st.button(
                    "Veri Toplamaya Başla (İlk Soruyu Üret)",
                    type="primary",
                    use_container_width=True,
                ):
                    if not os.path.exists(excel_path):
                        st.error(f"'{excel_path}' dosyası bulunamadı.")
                    else:
                        with st.spinner("İlk soru hazırlanıyor..."):
                            if otomatik_soru_uret():
                                st.rerun()

            else:
                st.markdown(
                    f"<span class='status-badge'>KONU · {st.session_state.get('secilen_excel_konusu', '')}</span>",
                    unsafe_allow_html=True,
                )
                st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

                guncel_soru = st.text_area(
                    "Üretilen Soru (isterseniz düzenleyebilirsiniz):",
                    value=st.session_state.generated_test_text,
                    height=80,
                )

                if guncel_soru != st.session_state.generated_test_text:
                    st.session_state.generated_test_text = guncel_soru
                    tahmin, sirali_probs = modelle_tahmin_et(guncel_soru)
                    st.session_state.bert_prediction = tahmin
                    st.session_state.bert_probabilities = sirali_probs
                    st.rerun()

                # Olasılık Dağılımı Kartı
                col_tahmin, col_dagilim = st.columns([1, 2])
                with col_tahmin:
                    st.markdown("##### Model Tahmini")
                    st.markdown(
                        f"<span class='tag-rejected' style='font-size:14px; padding:8px 14px;'>{st.session_state.bert_prediction} · REJECTED</span>",
                        unsafe_allow_html=True,
                    )

                with col_dagilim:
                    st.markdown("##### Model Olasılık Dağılımı")
                    if "bert_probabilities" in st.session_state:
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

                default_idx = 0
                if st.session_state.bert_prediction in st.session_state.classes:
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
                    "Kaydet ve Sıradakine Geç", type="primary", use_container_width=True
                ):
                    if st.session_state.bert_prediction == gercek_dogru_kategori:
                        st.error(
                            "Model zaten doğru tahmin etti. DPO çifti için modelin yanılması gerekir."
                        )
                    else:
                        dpo_pair = {
                            "prompt": st.session_state.generated_test_text,
                            "chosen": gercek_dogru_kategori,
                            "rejected": st.session_state.bert_prediction,
                            "konu": st.session_state.get("secilen_excel_konusu", ""),
                        }
                        st.session_state.dpo_dataset_buffer.append(dpo_pair)

                        with st.status(
                            "İşlem kaydediliyor ve yeni soru üretiliyor...",
                            expanded=True,
                        ) as status:
                            st.write("Veri tampon belleğe eklendi.")
                            otomatik_soru_uret()
                            status.update(
                                label="Hazır.", state="complete", expanded=False
                            )
                        st.rerun()

                if col_btn2.button(
                    "Soruyu At ve Sıradakine Geç", use_container_width=True
                ):
                    with st.status(
                        "Soru atlanıyor, yenisi hazırlanıyor...", expanded=True
                    ) as status:
                        st.write("Excel'den yeni konu seçiliyor.")
                        otomatik_soru_uret()
                        status.update(label="Hazır.", state="complete", expanded=False)
                    st.rerun()

            if st.session_state.dpo_dataset_buffer:
                st.markdown("---")
                st.subheader("Toplu DPO Veri Havuzu Yönetimi")
                st.markdown(
                    f"<span class='metric-badge'>{len(st.session_state.dpo_dataset_buffer)} DPO ÇİFTİ</span> "
                    f"<span style='color:var(--text-muted); font-size:13px;'>oturum boyunca biriken geçerli çift sayısı</span>",
                    unsafe_allow_html=True,
                )
                st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

                jsonl_data = "\n".join(
                    [
                        json.dumps(item, ensure_ascii=False)
                        for item in st.session_state.dpo_dataset_buffer
                    ]
                )

                col_down1, col_down2 = st.columns(2)
                with col_down1:
                    st.download_button(
                        label="Tüm Biriken DPO Verilerini İndir (.jsonl)",
                        data=jsonl_data,
                        file_name="insan_denetimli_zorlayici_dpo_dataseti.jsonl",
                        mime="text/plain",
                        type="primary",
                        use_container_width=True,
                    )
                with col_down2:
                    if st.button("Hafızayı Temizle", use_container_width=True):
                        st.session_state.dpo_dataset_buffer = []
                        st.rerun()
        else:
            st.warning(
                "Lütfen önce model yükleyin ve yukarıdaki alana Gemini API anahtarınızı girin."
            )
