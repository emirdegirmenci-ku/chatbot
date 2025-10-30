## PDF Dosyalarını Buraya Ekleyin

Bu klasöre Türkçe PDF dokümanlarınızı kopyalayın.

### Örnek:
```
data/pdf/
  ├── kullanici_dokumani.pdf
  ├── urun_katalogu.pdf
  └── sss.pdf
```

### PDF Formatı
- Türkçe metin içeren PDF dosyaları
- Taranmış görüntü değil, metin tabanlı PDF olmalı
- Dosya boyutu: Tercihen 50MB altı

### Yükleme
PDF'leri bu klasöre kopyaladıktan sonra şu komutu çalıştırın:

```bash
python scripts/ingest_pdf.py
```

Bu komut:
1. Tüm PDF'leri okur
2. 200-300 kelimelik parçalara böler
3. OpenAI API ile embedding üretir
4. ChromaDB vektör veritabanına kaydeder

### Not
İlk yükleme biraz zaman alabilir (PDF boyutuna bağlı).
OpenAI API'ye embedding istekleri gönderildiği için maliyet oluşur.

