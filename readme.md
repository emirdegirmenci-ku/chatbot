# Türkçe RAG Chatbot API Teknik Dokümantasyonu

Bu doküman, PDF dosyaları üzerinde "Retrieval-Augmented Generation" (RAG) tekniği ile çalışan Türkçe chatbot API'sinin kurulumu, yapılandırılması ve kullanımı hakkında detaylı bilgi içerir.

## Projeye Genel Bakış

Proje, belirli bir konu hakkındaki PDF dosyalarını bilgi kaynağı olarak kullanan bir sohbet robotu API'si sunar. Kullanıcı sorularına, yalnızca bu PDF'lerde bulunan bilgilere dayanarak yanıt üretir. Oturum (session) yönetimi sayesinde konuşma bağlamını takip edebilir ve her sohbeti daha sonra incelenmek üzere loglar.

Mimari, bir FastAPI backend sunucusu üzerine kuruludur ve API üzerinden kolayca entegre edilebilir.

---

## 1. Kurulum ve Yapılandırma

Projeyi yerel makinede çalıştırmak için aşağıdaki adımları izleyin.

### Ön Gereksinimler
- Python 3.10 veya üzeri
- `pip` paket yöneticisi

### Adım 1: Proje Dosyalarını Alın ve Sanal Ortam Oluşturun
```bash
# Projeyi klonlayın (eğer bir versiyon kontrol sisteminde ise)
# git clone <repository_url>
# cd chatbot-projesi

# Bir sanal ortam (virtual environment) oluşturun
python -m venv venv

# Sanal ortamı aktive edin
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### Adım 2: Gerekli Kütüphaneleri Yükleyin
Proje bağımlılıkları `requirements.txt` dosyasında listelenmiştir.
```bash
pip install -r requirements.txt
```

### Adım 3: Ortam Değişkenlerini (Environment Variables) Yapılandırın
Proje, hassas bilgileri ve yapılandırma ayarlarını `.env` dosyasından okur. `env.example` dosyasını kopyalayarak kendi `.env` dosyanızı oluşturun.

```bash
# Windows
copy env.example .env
# macOS / Linux
cp env.example .env
```
Ardından `.env` dosyasını açıp aşağıdaki alanları kendi bilgilerinizle doldurun:

- `AZURE_OPENAI_CHAT_ENDPOINT`: Azure OpenAI servisinizin chat modeli için uç nokta (endpoint) adresi.
- `AZURE_OPENAI_CHAT_KEY`: Azure OpenAI servisinizin API anahtarı.
- `AZURE_OPENAI_CHAT_DEPLOYMENT`: Azure üzerinde oluşturduğunuz chat modelinin dağıtım (deployment) adı.
- `AZURE_OPENAI_EMBED_ENDPOINT`: Azure OpenAI servisinizin embedding modeli için uç nokta adresi.
- `AZURE_OPENAI_EMBED_KEY`: Embedding modeli için API anahtarınız.
- `AZURE_OPENAI_EMBED_DEPLOYMENT`: Embedding modelinin dağıtım adı.
- `QDRANT_URL`: Vektör veritabanı Qdrant'ın adresi. (Varsayılan olarak bırakılabilir.)

Diğer ayarları (loglama konumu, zaman dilimi vb.) varsayılan olarak bırakabilirsiniz.

---

## 2. Veri Hazırlama (PDF Yükleme)

Chatbot'un cevap üreteceği PDF dosyalarını sisteme tanıtmanız (ingest) gerekir.

1.  Bilgi kaynağı olarak kullanılacak tüm `.pdf` dosyalarını `data/pdf/` klasörünün içine koyun.
2.  Aşağıdaki komutu çalıştırarak bu PDF'leri işleyin ve vektör veritabanına kaydedin:

```bash
python scripts/ingest_pdf.py
```
Bu işlem, PDF içeriğini parçalara ayırır, vektör temsillerini oluşturur ve Qdrant veritabanına kaydeder. Yeni bir PDF eklediğinizde bu komutu tekrar çalıştırmanız gerekir.

---

## 3. API Sunucusunu Çalıştırma

API sunucusunu başlatmak için projenin ana dizininde `run_backend.py` betiğini çalıştırın.

```bash
python run_backend.py
```
Alternatif olarak `uvicorn` komutunu da kullanabilirsiniz:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```
Sunucu başarıyla başladığında `Uvicorn running on http://0.0.0.0:8000` mesajını göreceksiniz.

---

## 4. API Endpoint Dokümantasyonu

Sunucu çalışırken API'ye istek atabilirsiniz. Aşağıda tüm endpoint'ler ve kullanımları detaylandırılmıştır.

**Not:** Windows PowerShell'de `curl` komutu farklı çalışır. Aşağıdaki örnekler `bash` veya `zsh` gibi Unix tabanlı terminaller için verilmiştir. Windows'ta [Git Bash](https://git-scm.com/downloads) veya WSL kullanmanız tavsiye edilir.

### 4.1. Yeni Sohbet Oturumu Oluşturma

Bir konuşma başlatmak için önce bir oturum ID'si almanız gerekir.

- **Endpoint:** `POST /api/session/create`
- **İstek (Request):** Body boş.
- **Yanıt (Response):** Oturum ID'sini içeren bir JSON nesnesi.
```json
{
    "session_id": "benzersiz_bir_uuid_degeri"
  }
  ```
- **Örnek `curl` komutu:**
  ```bash
  curl -X 'POST' 'http://127.0.0.1:8000/api/session/create' -H 'accept: application/json'
  ```

### 4.2. Sohbet Etme (Akışsız - Non-Streaming)

Tek seferde tam bir yanıt almak için bu endpoint'i kullanın.

- **Endpoint:** `POST /api/chat`
- **İstek (Request):**
  - `message` (zorunlu): Kullanıcının sorusu.
  - `session_id` (opsiyonel): Mevcut bir sohbete devam etmek için kullanılır. Eğer gönderilmezse, API otomatik olarak yeni bir oturum oluşturur.
```json
{
    "session_id": "önceki_adımdan_alınan_uuid",
    "message": "Yurt başvuruları ne zaman açılacak?"
  }
  ```
- **Yanıt (Response):** Cevabı, kaynakları ve diğer meta verileri içeren bir JSON nesnesi.
- **Örnek `curl` komutu:**
  ```bash
  curl -X 'POST' \
    'http://127.0.0.1:8000/api/chat' \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{
    "session_id": "9624b6c4-32a5-48f2-83b2-8d8a2dc1a93d",
    "message": "Ücreti ne kadar?"
  }'
  ```

### 4.3. Sohbet Etme (Akışlı - Streaming)

Cevabı yazılıyormuş gibi anlık olarak almak için bu endpoint kullanılır.

- **Endpoint:** `POST /api/chat/stream`
- **İstek (Request):** `/api/chat` ile aynı.
- **Yanıt (Response):** Her biri yeni bir satırda olan bir dizi JSON nesnesi.
  - `{"type": "token", "data": "parça"}`: Cevabın bir sonraki kelime/karakter parçasını içerir.
  - `{"type": "summary", ...}`: Akış bittiğinde, tam cevabı ve kaynakları içeren özet nesnesi.
- **Örnek `curl` komutu:**
  ```bash
  # --no-buffer parametresi çıktıyı anlık olarak görmek için önemlidir.
  curl --no-buffer -X 'POST' \
    'http://127.0.0.1:8000/api/chat/stream' \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{
    "message": "Yurt başvuruları ne zaman açılacak?"
  }'
  ```

### 4.4. Sohbet Oturumunu Sıfırlama/Silme

Bir oturumun sohbet geçmişini temizler.

- **Endpoint:** `POST /api/session/reset`
- **İstek (Request):**
```json
  {
    "session_id": "silinecek_oturum_uuid"
  }
  ```
- **Yanıt (Response):** İşlemin durumunu belirten bir JSON nesnesi.
- **Örnek `curl` komutu:**
  ```bash
  curl -X 'POST' \
    'http://127.0.0.1:8000/api/session/reset' \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{"session_id": "9624b6c4-32a5-48f2-83b2-8d8a2dc1a93d"}'
```

---

## 5. Komut Satırı Test Aracı (`chat_cli.py`)

Proje, API'yi kolayca test etmek için bir komut satırı aracı içerir.

- **Yeni bir sohbet başlatma (akışlı):**
  ```bash
  python scripts/chat_cli.py --message "Yurt başvuruları hakkında bilgi verir misin?" --stream
  ```
  Bu komutun çıktısında `Kullanılan oturum: <uuid>` şeklinde bir ID göreceksiniz.

- **Mevcut sohbete devam etme:**
  Bir önceki komuttan aldığınız `session_id`'yi kullanarak sohbete devam edebilirsiniz.
  ```bash
  python scripts/chat_cli.py --message "Başvuru ücreti nedir?" --stream --session-id <önceki_uuid>
  ```

---

## 6. Konuşma Logları

Her sohbet, `logs/conversations/` klasörü altında, oturum ID'si ile adlandırılmış `.jsonl` dosyalarına kaydedilir. Her satır, bir kullanıcı mesajını veya bir asistan cevabını temsil eder ve İstanbul saatine göre zaman damgası içerir.

**Örnek Log Satırı (`<session_id>.jsonl`):**
```json
{"timestamp": "2025-10-30T13:57:45.579623+03:00", "session_id": "9624b6c4-...", "role": "user", "content": "Yurt başvuruları ne zaman açılacak?"}
{"timestamp": "2025-10-30T13:57:48.123456+03:00", "session_id": "9624b6c4-...", "role": "assistant", "content": "Yurt başvuruları...", "sources": [...]}
```

---

## 7. İnteraktif API Dokümanları (Swagger UI)

API sunucusu çalışırken, `http://127.0.0.1:8000/docs` adresini tarayıcınızda açarak tüm endpoint'leri interaktif bir arayüz üzerinden test edebilirsiniz. Bu, API'yi keşfetmenin en kolay yoludur.

