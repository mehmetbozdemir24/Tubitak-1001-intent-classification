# Intent & Alignment Studio

Türkçe öğrenci mesajlarını niyet sınıflarına ayıran BERT modelini eğitip test etmeyi ve insan denetimli DPO hizalama verisi toplamayı tek arayüzden yapan Streamlit uygulaması.

Model: `dbmdz/bert-base-turkish-cased`

Sınıflar: `Bilgi_Karti`, `Anahtar_Kavram_Ozet`, `Kavramsal_Soru`, `Uyarlanabilir_Soru`, `Yanlis_Cevap_Aciklama`, `Kapsam_Disi`

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
4. **DPO Toplama** — Gemini API anahtarını gir, Excel'deki konu listesinden otomatik cümle ürettir, modelin yanlış tahmin ettiği örnekleri chosen/rejected çifti olarak onayla. Kayıtlar `dpo_dataset.jsonl` dosyasına eklenir.

## Repoda bulunmayan, yerelde hazır olması gereken dosyalar

Veri setleri, model ağırlıkları ve veri kaynakları `.gitignore` ile hariç tutulmuştur. Uygulamayı çalıştırmak için bunları proje kökünde kendiniz bulundurmanız gerekir:

| Dosya / klasör | Ne için gerekli |
| --- | --- |
| `Ders_konu_bilissel-seviye_öğrenme-çıktısı.xlsx` | DPO sekmesinde konu kaynağı. `Konu Adı` sütunu zorunludur. |
| Eğitim veri seti (`.jsonl`) | "Veri & Model" sekmesinden yüklenir. |
| `niyet_bert_model/` | Daha önce eğitilmiş modeli yüklemek için. Yoksa sekme 1'den eğitilir. |

Gemini API anahtarı koda yazılmaz; yalnızca DPO sekmesindeki şifre kutusundan girilir.

## Veri formatları

Eğitim veri seti — her satır bir örnek:

```json
{"text": "integral formüllerini özetler misin", "label": "Anahtar_Kavram_Ozet"}
```

DPO çıktısı (`dpo_dataset.jsonl`) — `chosen` insan kararı, `rejected` modelin hatalı tahmini:

```json
{"prompt": "...", "chosen": "Bilgi_Karti", "rejected": "Anahtar_Kavram_Ozet", "konu": "Hücre Zarından Madde Geçişi"}
```

## Eğitim çıktıları

Eğitim sonunda model ve tokenizer `niyet_bert_model/`, Trainer çıktıları `niyet_bert_results/` klasörüne yazılır. İkisi de repoya dahil edilmez.
