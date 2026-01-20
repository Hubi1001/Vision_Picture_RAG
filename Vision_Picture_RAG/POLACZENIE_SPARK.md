# 🚀 Konfiguracja VS Code z NVIDIA SPARK GPU

## Metoda 1: Remote Jupyter Kernel (ZALECANE - szybsze)

### Krok 1: Uzyskaj dane dostępowe do SPARK

W DGX Dashboard:
1. Kliknij **"Start"** w JupyterLab
2. Skopiuj URL JupyterLab (np. `https://spark.nvidia.com/lab?token=...`)
3. Znajdź token w URL lub w Settings

### Krok 2: Podłącz kernel w VS Code

1. W VS Code otwórz `metal_parts_rag.ipynb`
2. Kliknij na **"Select Kernel"** (prawy górny róg)
3. Wybierz **"Existing Jupyter Server"**
4. Wklej URL JupyterLab ze SPARK
5. Wybierz Python kernel ze SPARK

### Krok 3: Sprawdź połączenie

Uruchom komórkę:
```python
import torch
print(f"CUDA: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
```

---

## Metoda 2: Remote SSH (pełna integracja)

### Krok 1: Uzyskaj dane SSH do SPARK

Sprawdź w DGX Dashboard czy masz dostęp SSH:
- **Host:** `spark.nvidia.com` lub IP
- **Port:** zazwyczaj `22`
- **Username:** twoje konto NVIDIA
- **Klucz SSH:** jeśli wymagany

### Krok 2: Skonfiguruj SSH w VS Code

1. Naciśnij `F1` → wpisz **"Remote-SSH: Connect to Host"**
2. Wybierz **"Configure SSH Hosts"**
3. Edytuj plik config:

```ssh
Host nvidia-spark
    HostName [HOST_ZE_SPARK]
    User [TWOJ_USERNAME]
    Port 22
    IdentityFile ~/.ssh/id_rsa  # jeśli używasz klucza
```

4. Zapisz i połącz się: `F1` → **"Remote-SSH: Connect to Host"** → wybierz `nvidia-spark`

### Krok 3: Otwórz projekt zdalnie

Po połączeniu:
1. `File` → `Open Folder`
2. Upload projektu lub sklonuj z Git
3. Otwórz `metal_parts_rag.ipynb`
4. GPU ze SPARK będzie dostępne!

---

## Metoda 3: Jupyter Remote URI (jeśli Metoda 1 nie działa)

### Kroki:

1. **W DGX JupyterLab:** uruchom terminal i wykonaj:
```bash
jupyter notebook list
```
Skopiuj token.

2. **W VS Code:**
   - Otwórz Command Palette (`F1`)
   - Wpisz: **"Jupyter: Specify Jupyter Server for Connections"**
   - Wybierz **"Existing"**
   - Wklej: `http://[SPARK_IP]:8888/?token=[TOKEN]`

---

## ❓ Które dane potrzebujesz?

**Dla Metody 1 (zalecane):**
- [ ] URL JupyterLab ze SPARK (ze screenshota kliknij Start)

**Dla Metody 2 (zaawansowane):**
- [ ] Hostname/IP SPARK
- [ ] Username
- [ ] Klucz SSH lub hasło

**Podaj mi te dane, a pomogę skonfigurować połączenie!**

---

## 🔧 Rozwiązywanie problemów

### "Cannot connect to Jupyter server"
1. Sprawdź czy JupyterLab na SPARK jest uruchomiony (kliknij Start)
2. Sprawdź token w URL
3. Spróbuj z `https://` zamiast `http://`

### "SSH connection failed"
1. Sprawdź czy SPARK wymaga VPN
2. Sprawdź czy port 22 jest otwarty
3. Użyj klucza SSH zamiast hasła

### "Kernel died"
1. Za mało pamięci na SPARK - zamknij inne notebooki
2. Restart kernela: `Kernel` → `Restart`
