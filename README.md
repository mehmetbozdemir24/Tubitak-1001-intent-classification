# Intent & Alignment Studio

Türkçe niyet (intent) sınıflandırma için BERT modeli eğitip test etmeyi ve insan denetimli DPO hizalama verisi toplamayı tek arayüzden yapan Streamlit uygulaması.

## Kurulum

```
pip install -r requirements.txt
```

## Çalıştırma

```
streamlit run tubitak-1001-niyet-tespiti.py
```

## Kullanım

Uygulama 4 sekmeden oluşur, sırayla kullanılır:

1. **Veri & Model** — JSONL formatında etiketli veri yükleyip yeni bir BERT modeli eğit, ya da daha önce eğitilmiş bir modeli klasör yolundan yükle.
2. **Performans** — eğitilen/yüklenen modelin accuracy, F1, precision, recall değerlerini ve confusion matrix'ini görüntüle.
3. **Manuel Test** — model yüklüyken serbest metin gir, tahmin edilen niyeti ve sınıf olasılık dağılımını gör.
4. **DPO Toplama** — Gemini API anahtarını gir, Excel'deki konu listesinden otomatik soru ürettir, modelin yanlış tahmin ettiği örnekleri chosen/rejected çifti olarak onaylayıp `.jsonl` formatında indir.

## Gerekli dosyalar

- `Ders_konu_bilissel-seviye_öğrenme-çıktısı.xlsx` — DPO sekmesinde otomatik soru üretimi için konu kaynağı.
- Gemini API anahtarı — sadece DPO sekmesinde, uygulama içinden girilir; koda yazılmaz.

## Not

`niyet_bert_model/` (eğitilen model ağırlıkları) `.gitignore` ile hariç tutulmuştur; büyük boyutları nedeniyle repoya dahil edilmez.
