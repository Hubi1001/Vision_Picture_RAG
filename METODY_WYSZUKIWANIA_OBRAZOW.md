# Metody Wyszukiwania Obrazów Podobnych (na Podstawie Metadanych)

## 1. Wstęp Teoretyczny

Wyszukiwanie obrazów podobnych to kluczowy problem w wizji komputerowej i informatyce. Istnieją dwa główne podejścia:

### 1.1 **Content-Based Image Retrieval (CBIR)** vs **Metadata-Based Search**

**CBIR** (wyszukiwanie na podstawie zawartości) analizuje rzeczywistą zawartość obrazu (piksele, kolory, tekstury, kształty), podczas gdy **metadata-based search** opiera się na ręcznie dodanych słowach kluczowych i tagach.

**Zalety CBIR:**
- Nie wymaga ręcznego tagowania
- Może znaleźć obrazy z różnymi opisami, ale podobną zawartością
- Automatyczne, skalowalne dla dużych baz danych

**Zalety metadata-based search:**
- Szybkie wyszukiwanie
- Nie wymaga wydajności obliczeniowej
- Intuicyjne dla użytkownika

---

## 2. Kluczowe Metody Wyszukiwania Obrazów

### 2.1 **Query By Image Content (QBIC)** — historycznie pierwszy system

- **Okreś:** Opracowany przez IBM w 1992r.
- **Metodologia:** Analizuje kolory, tekstury, kształty
- **Dostęp:** Użytkownik podaje przykładowy obraz → system szuka podobnych

### 2.2 **Query By Example (QBE)**

- Użytkownik dostarcza przykładowy obraz lub rysuje przybliżenie
- System znajduje obrazy z podobnymi elementami
- **Implementacja:** wektory cech (feature vectors) → porównanie odległości

### 2.3 **VisualRank** (Google)

- Stosuje PageRank do wyszukiwania obrazów
- Analizuje linki i kontekst wizualny
- Używane w Google Image Search

### 2.4 **Semantic Retrieval**

- "Znaleźć zdjęcia Abrahama Lincolna"
- Wymaga rozpoznawania wyższych koncepcji
- Często wymaga feedbacku użytkownika (man-in-the-loop)

### 2.5 **Relevance Feedback**

- Iteracyjne udoskonalanie wyszukiwania
- Użytkownik oznacza wyniki jako "relevant", "not relevant", "neutral"
- System uczy się z czasem

### 2.6 **Machine Learning & Deep Learning**

- Embeddingi nauczane na dużych datasetach (np. ImageNet)
- Sieci neuronowe: CNNs, Vision Transformers
- **Współczesne podejście:** Siamese networks, metric learning

---

## 3. Techniki Ekstrakcji Cech (Features)

### 3.1 **Kolor (Color)**
- **Histogram kolorów:** proporcje pikseli o konkretnych wartościach
- **Segregacja czasowa:** gdzie w obrazie znajduje się dany kolor
- **Zaleta:** niezawisne od rozmiaru i obrotu obrazu

### 3.2 **Tekstura (Texture)**
- **Co-occurrence Matrix (GLCM):** analiza pary pikseli
- **Laws Texture Energy:** modele energii tekstury
- **Wavelet Transform:** wieloskalowa analiza
- **Orthogonal Transforms:** dyskretne momenty Chebysheva

### 3.3 **Kształt (Shape)**
- Wydzielenie konturów (edge detection, segmentacja)
- **Deskryptory:** Fourier Transform, Moment Invariant
- Muszą być niezmienne względem translacji, rotacji, skalowania

### 3.4 **Embeddingi (nowoczesne)**
- Sieci neuronowe trenowane end-to-end
- Wektory o wysokiej wymiarowości (512-2048D)
- **Przykłady:** CLIP, ResNet50, Vision Transformers
- **Metryka:** cosine similarity, L2 distance

---

## 4. Miary Odległości Obrazów

| Metryka | Zastosowanie | Uwagi |
|---------|-------------|-------|
| **Euclidean Distance** | Standardowe porównanie wektorów | Wrażliwa na skalę |
| **Cosine Similarity** | Embeddingi, znormalizowane cechy | Niezawisna od amplitudy |
| **Manhattan Distance** | Szybsze obliczenia niż Euclidean | Mniej wrażliwa na outliers |
| **Hamming Distance** | Binarne deskryptory | Szybkie porównania |
| **Chi-Square Distance** | Histogramy kolorów | Probabilistyczne podejście |

---

## 5. Zastosowania Praktyczne

### Przemysł / Inżynieria
- **Kontrola jakości:** wyszukiwanie wadliwych części metalowych
- **Katalogi części:** szybkie znalezienie odpowiednika
- **Metrologia:** porównanie wymiarów na zdjęciach

### Medycyna
- Diagnostyka: znalezienie podobnych przypadków rentgenów
- Archiwa: szybkie wyszukiwanie historycznych przypadków

### e-Commerce
- Reverse image search (szukaj przez zdjęcie)
- Rekomendacje produktów

### Bezpieczeństwo
- Rozpoznawanie twarzy
- Detekcja obiektów podejrzanych

---

## 6. Nowoczesne Podejście: RAG dla Obrazów

**RAG = Retrieval-Augmented Generation**

1. **Retriever:** Znajdź podobne obrazy na podstawie zapytania (tekst lub obraz)
2. **Augmentation:** Dodaj metadane, opisy, historeję znalezionych obrazów
3. **Generation:** Wygeneruj odpowiedź/raport z znalezionymi obrazami

### Architektura:
```
Użytkownik pisze pytanie
        ↓
Konwertuj pytanie na embedding (tekst→wektor)
        ↓
Wyszukaj podobne obrazy w bazie (cosine similarity)
        ↓
Pobierz metadane znalezionych obrazów
        ↓
Zbuduj kontekst (opis + metatagi + historia)
        ↓
Wyślij do LLM z instrukcją
        ↓
LLM generuje odpowiedź + rekomenduje obrazy
```

---

## 7. Dane Treningowe: Części Metalowe

### Źródła Zdjęć w Internecie:

1. **Katalogi Producentów**
   - McMaster-Carr: mcmaster.com (śruby, złącza)
   - Digi-Key, Mouser Electronics: elektronika
   - LinParts, RoboticShop: części mechaniczne

2. **Open Datasets**
   - **ImageNet:** ogólne kategorie, brak specjalizacji
   - **Open Images:** Google's large-scale dataset
   - **Parts-in-Context (PIC):** części w kontekście
   - **ScanNet:** sceny 3D z partiami

3. **Zdjęcia Indywidualne**
   - Zdjęcia zrobione własnie o kontrolowanym oświetleniu
   - Standaryzowana orientacja (front, boki, 3D)
   - EXIF metadata: data, aparat, ustawienia ISO

### Rekomendowane Metadane dla Części Metalowych:
```json
{
  "part_id": "SCR-M8-1.25-20",
  "description": "Śruba sześciokątna M8",
  "material": "Stal nierdzewna A2-70",
  "dimensions": {
    "diameter_mm": 8.0,
    "length_mm": 20.0,
    "pitch_mm": 1.25
  },
  "tags": ["śruba", "metalowa", "nierdzewna", "heksagonalna"],
  "category": "fasteners",
  "image_path": "parts_db/fasteners/scr_m8_20.jpg",
  "capture_conditions": {
    "lighting": "LED ring light",
    "background": "white",
    "angle_degrees": 45
  }
}
```

---

## 8. Algorytm Rekomendowany dla RAG Obrazów

### **Hybrid Approach:**

```
1. METADATA FILTER (szybkie sito)
   - Wyszukaj kategorie: fasteners, bearings, springs...
   - Filtruj po material, size range
   
2. VISUAL FEATURE EXTRACTION
   - CNN embedding (np. ResNet50 pretrainowany)
   - Kolory, tekstura, histogramy
   
3. VECTOR SIMILARITY SEARCH
   - Cosine similarity w space embeddingów
   - Top-K nearest neighbors
   
4. RERANKING (opcjonalnie)
   - Dodatkowe warunki: rozmiar, oświetlenie
   - Ranking hybrydowy: metadata + visual
   
5. CONTEXT BUILDING
   - Metadane znalezionych części
   - Historia porównań
   - Dokumentacja (CAD, specyfikacja)
   
6. LLM GENERATION
   - Wygeneruj raport o znalezionych częściach
   - Rekomenduj alternatywy
```

---

## 9. Podsumowanie

| Aspekt | Rekomendacja |
|--------|-------------|
| **Metoda wyszukiwania** | Hybrid (metadata + embeddingi) |
| **Ekstrakcja cech** | Pretrainowana CNN + manual features |
| **Metryka odległości** | Cosine similarity |
| **Baza danych** | Vector DB (Weaviate, Pinecone) lub PostgreSQL+pgvector |
| **LLM** | TinyLlama lub Qwen dla CPU-only |
| **Framework** | RAG (retriever + LLM) |

---

## Referencje

- **Wikipedia CBIR:** Content-based Image Retrieval
- **IBM QBIC:** Flickner et al., 1995
- **VisualRank:** Jing & Baluja, Google, 2008
- **ArXiv Computer Vision:** Ostatnie artykuły na cs.CV

---

*Dokument przygotowany: 10.12.2025*
*Kontekst: Implementacja RAG-owego systemu wyszukiwania obrazów części metalowych*
