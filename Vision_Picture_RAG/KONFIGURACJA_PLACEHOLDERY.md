# 📋 Konfiguracja i Placeholdery - Vision Picture RAG

## ✅ Optymalizacje dla GitHub Codespace (CPU)

Notatnik został dostosowany do działania na CPU w środowisku GitHub Codespace:

### Zmiany wprowadzone:
- ✅ Model LLM zmieniony z `Qwen/Qwen2.5-1.5B-Instruct` (4-6GB) na `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (~2GB)
- ✅ Domyślna baza danych: SQLite (bez dodatkowej konfiguracji)
- ✅ Model embeddings: `paraphrase-multilingual-MiniLM-L12-v2` (działa sprawnie na CPU)
- ✅ Dodano obsługę błędów przy ładowaniu modeli (try-except)
- ✅ Dodano przykładowe dane testowe (syntetyczne)

---

## 🔧 PLACEHOLDERY DO UZUPEŁNIENIA

### [PLACEHOLDER 1] - Połączenie z bazą danych
**Lokalizacja:** Komórka 3 (sekcja: Setup i konfiguracja)

```python
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./vision.db")
```

**Co zrobić:**
- **Domyślnie:** Użyj SQLite (nic nie zmieniaj) - działa od razu w Codespace
- **Dla PostgreSQL:** 
  1. Ustaw zmienną środowiskową:
     ```bash
     export DATABASE_URL="postgresql://user:password@localhost:5432/vision_db"
     ```
  2. Lub w Codespace: Settings → Secrets → dodaj `DATABASE_URL`
  3. Format: `postgresql://[user]:[password]@[host]:[port]/[database_name]`

**Przykład produkcyjny:**
```python
DATABASE_URL = "postgresql://vision_user:TajneHaslo123@db.example.com:5432/vision_production"
```

---

### [PLACEHOLDER 2] - Ścieżka do notatek tekstowych
**Lokalizacja:** Komórka 26 (sekcja: Przykładowe użycie)

```python
# notes_root = Path("./notatki")  # Zmień na swoją ścieżkę
# index_notes_from_folder(db, notes_root)
```

**Co zrobić:**
1. Utwórz folder z notatkami tekstowymi - podsumowaniami materiałów (pliki `.md` lub `.txt`)
2. Nazwij pliki wg konwencji: `TOPIC-OPTIMIZATION_notes.md` (temat na początku)
3. Treść: Twoje własne podsumowania, kluczowe punkty, linki źródłowe
4. Zmień ścieżkę na właściwą i odkomentuj linijki

**Przykład:**
```python
notes_root = Path("/workspaces/Vision_Picture_RAG/notatki")
index_notes_from_folder(db, notes_root)
```

**Struktura przykładowa:**
```
notatki/
├── TOPIC-OPTIMIZATION_gradient_descent.md
│   # Zawartość: "Gradient descent - algorytm optymalizacji..."
│   # Źródło: Coursera - Andrew Ng ML course
├── TOPIC-NEURAL-NETS_backpropagation.md
│   # Zawartość: "Backpropagation wyjaśnienie..."
│   # Źródło: 3Blue1Brown YouTube
└── TOPIC-PROBABILITY_distributions.md
    # Zawartość: "Rozkłady statystyczne - cheat sheet..."
    # Źródło: r/datascience infografika
```

**Wskazówki:**
- Dodawaj linki do źródeł w notatkach
- Używaj formatowania Markdown dla czytelności
- Grupuj notatki tematycznie (nie chronologicznie)

---

### [PLACEHOLDER 3] - Ścieżka do zdjęć notatek z internetu
**Lokalizacja:** Komórka 26 (sekcja: Przykładowe użycie)

```python
# images_root = Path("./obrazy")  # ← Tutaj są zdjęcia notatek z internetu
# index_images_from_folder(db, images_root, default_project_id="ML-COURSE-2025")
```

**Co zrobić:**
1. W folderze `obrazy` umieść materiały pobrane z internetu (`.jpg`, `.png`, `.jpeg`, `.webp`)
2. Nazwij pliki opisowo, np: `lecture_03_slide_15.png`, `youtube_screenshot_backprop.jpg`
3. System automatycznie rozpozna typy na podstawie nazw:
   - `lecture*` / `wyklad*` → tag "wykład"
   - `slide*` / `slajd*` → tag "slajd"
   - `chart*` / `wykres*` / `graph*` → tag "wykres"
   - `infographic*` / `schema*` → tag "infografika"
   - `screenshot*` → tag "screenshot"
   - `youtube*` / `yt*` → tag "youtube"
   - `arxiv*` → tag "arxiv"
   - `pdf*` → tag "pdf"
4. Organizuj w podfoldery według tematów/kursów

**Przykład:**
```python
images_root = Path("/workspaces/Vision_Picture_RAG/obrazy")
index_images_from_folder(db, images_root, default_project_id="ML-COURSE-2025")
```

**Struktura przykładowa:**
```
obrazy/
├── machine_learning/
│   ├── lecture_03_slide_15.png          ← Slajd z PDF wykładu
│   ├── youtube_3blue1brown_backprop.jpg ← Screenshot z YouTube
│   └── coursera_gradient_descent.png    ← Materiał z Coursera
├── deep_learning/
│   ├── arxiv_paper_fig3.png             ← Wykres z artykułu arXiv
│   └── infographic_neural_nets.jpg      ← Infografika z internetu
└── statistics/
    ├── chart_distributions.png          ← Wykres rozkładów
    └── slide_hypothesis_testing.jpg     ← Slajd o testach hipotez
```

**Typy materiałów z internetu:**
- 📄 **Strony z PDF-ów** (wykłady, artykuły, podręczniki)
- 🎥 **Screenshoty YouTube** (wykłady online, tutoriale)
- 📊 **Wykresy/diagramy** (z artykułów, blogów naukowych)
- 🖼️ **Infografiki** (Reddit, Medium, Twitter/X)
- 📑 **Slajdy** (prezentacje z kursów online)
- 📚 **Materiały z arxiv** (rysunki z publikacji naukowych)

---

### [PLACEHOLDER 4] - ID projektu
**Lokalizacja:** Komórka 26 (sekcja: Przykładowe użycie)

```python
index_images_from_folder(db, images_root, default_project_id="PROJ-001")
```

**Co zrobić:**
Zmień `"PROJ-001"` na unikalny identyfikator swojego projektu

**Przykłady:**
```python
default_project_id="MATERIALS-TESTING-2025"
default_project_id="LAB-RESEARCH-Q1"
default_project_id="THESIS-EXPERIMENT-01"
```

---

### [PLACEHOLDER 5] - Model LLM
**Lokalizacja:** Komórka 14 (sekcja: Modele embeddingi i LLM)

```python
LLM_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # ✅ REKOMENDOWANE dla Codespace
```

**Co zrobić:**
- **Domyślnie:** Zostaw `TinyLlama` (najlepsze dla Codespace CPU)
- **Jeśli masz więcej RAM:** Możesz zmienić na większy model

**Dostępne opcje:**

| Model | RAM | Język | Uwagi |
|-------|-----|-------|-------|
| `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | ~2GB | EN/wielojęzyczny | ✅ **Rekomendowane** dla Codespace |
| `gpt2` | ~500MB | Angielski | Backup, mały |
| `distilgpt2` | ~350MB | Angielski | Najlżejszy |
| `Qwen/Qwen2.5-1.5B-Instruct` | ~4-6GB | Wielojęzyczny | Może przekroczyć limit RAM |
| `microsoft/phi-2` | ~3GB | Angielski | Dobra jakość |

**Przykład zmiany:**
```python
LLM_MODEL_NAME = "microsoft/phi-2"  # Jeśli masz 8GB+ RAM
```

---

## 🚀 Szybki Start (bez własnych danych)

Jeśli chcesz przetestować system bez przygotowywania danych:

1. **Zainstaluj pakiety** (Komórka 2):
   ```
   Uruchom → Restart Kernel
   ```

2. **Uruchom komórki 3-13** (setup, modele, funkcje)

3. **W komórce 26 odkomentuj sekcję "OPCJA A: DANE TESTOWE"**:
   ```python
   # Odkomentuj linie od test_img1 do ostatniego print
   ```

4. **Uruchom komórkę 26** - system stworzy syntetyczne dane i przetestuje RAG

---

## 🔐 Zmienne środowiskowe (opcjonalne)

Jeśli używasz PostgreSQL lub Hugging Face z prywatnymi modelami:

### W GitHub Codespace:
```bash
# Dodaj do ~/.bashrc lub jako Secret w ustawieniach Codespace
export DATABASE_URL="postgresql://user:pass@host:5432/db"
export HF_TOKEN="hf_your_token_here"  # Tylko dla prywatnych modeli HF
```

### Lokalnie:
```bash
# Stwórz plik .env w katalogu projektu
echo 'DATABASE_URL=postgresql://user:pass@localhost:5432/vision_db' > .env
echo 'HF_TOKEN=hf_your_token_here' >> .env
```

Następnie załaduj w notebooku:
```python
from dotenv import load_dotenv
load_dotenv()
```

---

## 📊 Testowanie po konfiguracji

Po uzupełnieniu placeholderów:

1. **Test połączenia z bazą** (Komórka 6):
   ```
   Powinno wypisać: "Połączenie z bazą działa, SessionLocal OK."
   ```

2. **Test indeksowania** (Komórka 12):
   ```
   Powinno pokazać liczbę obrazów i notatek
   ```

3. **Test embeddingów** (Komórka 15):
   ```
   "Funkcja get_embedding jeszcze nie jest zdefiniowana..." = OK
   ```

4. **Test RAG** (Komórka 24):
   ```
   Jeśli masz dane: zwróci odpowiedź i image_ids
   Jeśli brak danych: pusta lista obrazów
   ```

---

## ⚠️ Częste problemy

### Problem: Kernel się wykrzacza przy ładowaniu LLM
**Rozwiązanie:**
```python
# W komórce 14 zmień model na lżejszy:
LLM_MODEL_NAME = "gpt2"  # lub "distilgpt2"
```

### Problem: Brak obrazów w bazie
**Rozwiązanie:**
- Sprawdź ścieżkę w `images_root`
- Upewnij się, że obrazy mają rozszerzenie `.jpg`
- Odkomentuj i uruchom `index_images_from_folder()`

### Problem: Import Error dla pgvector
**Rozwiązanie:**
- Jeśli używasz SQLite - ignoruj (to normalne)
- Jeśli PostgreSQL - zainstaluj: `pip install pgvector`

---

## 📝 Podsumowanie checklist

- [ ] [PLACEHOLDER 1] Skonfigurowano DATABASE_URL (lub zostawiono SQLite)
- [ ] [PLACEHOLDER 2] Ustawiono ścieżkę do notatek `notes_root`
- [ ] [PLACEHOLDER 3] Ustawiono ścieżkę do obrazów `images_root`
- [ ] [PLACEHOLDER 4] Zmieniono `default_project_id` na właściwy
- [ ] [PLACEHOLDER 5] Wybrano odpowiedni model LLM dla dostępnego RAM
- [ ] Zainstalowano pakiety (komórka 2) i zrestartowano kernel
- [ ] Uruchomiono komórki 3-13 (setup)
- [ ] Zindeksowano dane lub użyto danych testowych
- [ ] Przetestowano funkcję `answer_question()`

---

Powodzenia! 🎉
