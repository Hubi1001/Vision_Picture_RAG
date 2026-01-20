# ℹ️ Informacje o GPU w Twoim systemie

## Status sprzętu

**Karta graficzna:** Intel(R) UHD Graphics  
**Typ:** Zintegrowana karta Intel  
**CUDA:** ❌ Nie obsługuje (tylko karty NVIDIA)  
**PyTorch:** 2.9.1+cpu (wersja CPU-only)

---

## 🔍 Dlaczego nie działa GPU w PyTorch?

CUDA (Compute Unified Device Architecture) to technologia **tylko dla kart NVIDIA**:
- ✅ Karty NVIDIA: GeForce RTX/GTX, Tesla, Quadro
- ❌ Karty Intel: UHD Graphics, Iris Xe
- ❌ Karty AMD: Radeon

Twoja Intel UHD Graphics **nie obsługuje CUDA**, więc PyTorch musi działać na CPU.

---

## ✅ Co możesz zrobić?

### Opcja 1: Używaj CPU (zalecane)
Projekt **działa poprawnie na CPU**, tylko wolniej:
- ✅ Wszystkie funkcje działają
- ✅ Nie wymaga zmian w kodzie
- ⏱️ Wolniejsze obliczenia (np. 10s zamiast 1s)

**Aktualna konfiguracja:**
```python
DEVICE = "cpu"  # Automatycznie wykryte
```

### Opcja 2: Intel GPU przez DirectML (eksperymentalne)

Intel GPU może być używane przez **DirectML** (Windows):

```bash
pip install torch-directml
```

Potem w kodzie:
```python
import torch_directml
DEVICE = torch_directml.device()
```

⚠️ **Uwaga:** DirectML jest eksperymentalne i może nie działać ze wszystkimi modelami.

### Opcja 3: Kup kartę NVIDIA (sprzętowe)

Jeśli potrzebujesz GPU do ML:
- **Budget:** NVIDIA GTX 1660 Super (6GB VRAM) - ~800 PLN
- **Średnia:** NVIDIA RTX 3060 (12GB VRAM) - ~1500 PLN  
- **Wysoka:** NVIDIA RTX 4070 (12GB VRAM) - ~2500 PLN

### Opcja 4: Użyj chmury

**Google Colab** (darmowe GPU):
- 🆓 Darmowe Tesla T4 (15GB VRAM)
- ⏱️ Limit: 12h sesji
- 📤 Upload kodu i danych
- 🔗 https://colab.research.google.com

**Paperspace Gradient** (płatne):
- 💰 Od $0.45/h (RTX 4000)
- ⚡ Szybkie GPU
- 💾 Stała przestrzeń dyskowa

---

## 📊 Porównanie wydajności

| Operacja | Intel UHD (CPU) | NVIDIA RTX 3060 (GPU) |
|----------|-----------------|----------------------|
| CLIP embedding | ~200ms | ~15ms |
| Phi-3 generacja | ~15s | ~800ms |
| Batch 32 obrazów | ~6.4s | ~300ms |

---

## 🎯 Rekomendacja dla Twojego projektu

**Dla testów i nauki:** Używaj CPU ✅
- Projekt działa
- Nie wymaga inwestycji
- Wystarczy do prototypowania

**Dla produkcji:** Rozważ Google Colab lub GPU w chmurze
- Szybkie przetwarzanie
- Bez kosztów sprzętu
- Łatwa skalacja

---

## 🔧 Jak uruchomić projekt na CPU

Wszystko jest już skonfigurowane! Notebook automatycznie wykryje brak GPU i użyje CPU:

```python
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Na Twoim komputerze: DEVICE = "cpu"
```

Uruchom normalnie wszystkie komórki - projekt będzie działać poprawnie.

---

## ❓ Pytania i odpowiedzi

**Q: Czy mogę w ogóle używać Intel GPU?**  
A: Tak, przez DirectML, ale wsparcie jest ograniczone i eksperymentalne.

**Q: Dlaczego PyTorch zainstalował się bez CUDA?**  
A: PyTorch automatycznie wykrył brak karty NVIDIA i zainstalował wersję CPU.

**Q: Czy projekt w ogóle zadziała bez GPU?**  
A: TAK! Wszystko działa, tylko wolniej. GPU to tylko przyspieszenie.

**Q: Czy mogę emulować CUDA na Intel?**  
A: Nie. CUDA to zamknięta technologia NVIDIA.

---

## 📚 Dodatkowe zasoby

- [PyTorch CPU vs GPU](https://pytorch.org/get-started/locally/)
- [Intel Extension for PyTorch](https://github.com/intel/intel-extension-for-pytorch)
- [DirectML Documentation](https://learn.microsoft.com/en-us/windows/ai/directml/dml)
- [Google Colab Tutorial](https://colab.research.google.com/)
