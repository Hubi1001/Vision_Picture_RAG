#!/usr/bin/env python3
"""
Moduł do wyszukiwania ofert części metalowych w internecie.
Obsługuje wyszukiwanie przez Google, Allegro, Amazon i inne platformy e-commerce.
"""

from __future__ import annotations

import json
import urllib.parse
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime
import time

# Opcjonalne importy do web scrapingu i API
try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    from bs4 import BeautifulSoup
    HAS_BEAUTIFULSOUP = True
except ImportError:
    HAS_BEAUTIFULSOUP = False


@dataclass
class OnlineOffer:
    """Model oferty znalezionej w internecie."""
    title: str
    price: Optional[str] = None
    currency: str = "PLN"
    url: str = ""
    seller: str = ""
    source: str = ""  # "allegro", "amazon", "ebay", "aliexpress", itp.
    availability: str = "unknown"
    rating: Optional[float] = None
    description: str = ""
    image_url: str = ""
    timestamp: str = ""
    relevance_score: float = 0.0  # 0-1, jak bardzo pasuje do zapytania

    def to_dict(self) -> Dict:
        """Konwertuj na słownik."""
        return {
            "title": self.title,
            "price": self.price,
            "currency": self.currency,
            "url": self.url,
            "seller": self.seller,
            "source": self.source,
            "availability": self.availability,
            "rating": self.rating,
            "description": self.description,
            "image_url": self.image_url,
            "timestamp": self.timestamp,
            "relevance_score": self.relevance_score,
        }


class OfferFinder:
    """Klasa do wyszukiwania ofert w internecie."""

    def __init__(self, use_proxies: bool = False):
        """
        Inicjalizacja findera ofert.
        
        Args:
            use_proxies: Czy używać proxy (ogranicza blokady)
        """
        self.use_proxies = use_proxies
        self.session = self._create_session() if HAS_REQUESTS else None
        self.offers_cache = {}

    def _create_session(self) -> requests.Session:
        """Utwórz sesję z retry strategy."""
        session = requests.Session()
        retry = Retry(total=3, backoff_factor=0.5)
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                )
            }
        )
        return session

    def search_allegro(
        self, query: str, max_results: int = 5
    ) -> List[OnlineOffer]:
        """
        Wyszukaj na Allegro.pl
        
        Args:
            query: Zapytanie wyszukiwania (np. "śruba M8" lub "part_id")
            max_results: Maksymalna liczba wyników
            
        Returns:
            Lista ofert z Allegro
        """
        if not HAS_REQUESTS:
            return []

        offers = []
        try:
            # URL Allegro
            base_url = "https://allegro.pl/listing"
            params = {"string": query}
            url = f"{base_url}?{urllib.parse.urlencode(params)}"

            # Allegro blokuje boty - wróć linki do manualnego przeglądania
            offer = OnlineOffer(
                title=f"[Allegro] {query}",
                url=url,
                source="allegro",
                seller="allegro.pl",
                description=f"Wyszukaj ręcznie na Allegro: {query}",
                timestamp=datetime.now().isoformat(),
            )
            offers.append(offer)
        except Exception as e:
            print(f"⚠ Błąd wyszukiwania Allegro: {e}")

        return offers[:max_results]

    def search_amazon(
        self, query: str, max_results: int = 5
    ) -> List[OnlineOffer]:
        """Wyszukaj na Amazon.com (EUR wersja)."""
        if not HAS_REQUESTS:
            return []

        offers = []
        try:
            base_url = "https://www.amazon.eu/s"
            params = {"k": query}
            url = f"{base_url}?{urllib.parse.urlencode(params)}"

            offer = OnlineOffer(
                title=f"[Amazon] {query}",
                url=url,
                source="amazon",
                seller="amazon.eu",
                description=f"Wyszukaj na Amazon EU: {query}",
                timestamp=datetime.now().isoformat(),
            )
            offers.append(offer)
        except Exception as e:
            print(f"⚠ Błąd wyszukiwania Amazon: {e}")

        return offers[:max_results]

    def search_aliexpress(
        self, query: str, max_results: int = 5
    ) -> List[OnlineOffer]:
        """Wyszukaj na AliExpress."""
        if not HAS_REQUESTS:
            return []

        offers = []
        try:
            base_url = "https://www.aliexpress.com/wholesale"
            params = {"SearchText": query}
            url = f"{base_url}?{urllib.parse.urlencode(params)}"

            offer = OnlineOffer(
                title=f"[AliExpress] {query}",
                url=url,
                source="aliexpress",
                seller="aliexpress.com",
                description=f"Części z Chin - wyszukaj na AliExpress: {query}",
                timestamp=datetime.now().isoformat(),
            )
            offers.append(offer)
        except Exception as e:
            print(f"⚠ Błąd wyszukiwania AliExpress: {e}")

        return offers[:max_results]

    def search_ebay(self, query: str, max_results: int = 5) -> List[OnlineOffer]:
        """Wyszukaj na eBay."""
        if not HAS_REQUESTS:
            return []

        offers = []
        try:
            base_url = "https://www.ebay.com/sch/i.html"
            params = {"_nkw": query}
            url = f"{base_url}?{urllib.parse.urlencode(params)}"

            offer = OnlineOffer(
                title=f"[eBay] {query}",
                url=url,
                source="ebay",
                seller="ebay.com",
                description=f"Części na eBay - nowe i używane: {query}",
                timestamp=datetime.now().isoformat(),
            )
            offers.append(offer)
        except Exception as e:
            print(f"⚠ Błąd wyszukiwania eBay: {e}")

        return offers[:max_results]

    def search_google(self, query: str, max_results: int = 5) -> List[OnlineOffer]:
        """
        Ogólne wyszukiwanie Google dla części i ofert.
        
        Zwraca bezpośrednie linki do wyszukiwania (bez scrapingu, aby nie naruszać ToS).
        """
        offers = []
        
        # Wyszukiwanie na różnych domenach
        search_queries = [
            (f"{query} cena", "https://www.google.com/search?q="),
            (f"{query} buy online", "https://www.google.com/search?q="),
            (f"{query} shop", "https://www.google.com/search?q="),
        ]

        for search_term, base_url in search_queries:
            try:
                url = base_url + urllib.parse.quote(search_term)
                offer = OnlineOffer(
                    title=f"[Google] Wyniki dla: {search_term}",
                    url=url,
                    source="google",
                    seller="google.com",
                    description=f"Wyszukaj w Google: {search_term}",
                    timestamp=datetime.now().isoformat(),
                )
                offers.append(offer)
            except Exception as e:
                print(f"⚠ Błąd tworzenia linku Google: {e}")

        return offers[:max_results]

    def search_all_platforms(
        self,
        query: str,
        platforms: Optional[List[str]] = None,
        max_results_per_platform: int = 3,
    ) -> List[OnlineOffer]:
        """
        Wyszukaj na wszystkich lub wybranych platformach jednocześnie.
        
        Args:
            query: Zapytanie wyszukiwania
            platforms: Lista platform ("allegro", "amazon", "ebay", "aliexpress", "google")
                      Jeśli None, przeszukuje wszystkie
            max_results_per_platform: Ile wyników z każdej platformy
            
        Returns:
            Połączona lista ofert ze wszystkich platform
        """
        if platforms is None:
            platforms = ["allegro", "amazon", "ebay", "aliexpress", "google"]

        all_offers = []

        print(f"\n🔍 Wyszukiwanie dla: {query}")
        print(f"📱 Platformy: {', '.join(platforms)}\n")

        # Allegro
        if "allegro" in platforms:
            print("  📌 Allegro...", end=" ")
            offers = self.search_allegro(query, max_results_per_platform)
            all_offers.extend(offers)
            print(f"✓ ({len(offers)})")
            time.sleep(0.5)

        # Amazon
        if "amazon" in platforms:
            print("  📦 Amazon...", end=" ")
            offers = self.search_amazon(query, max_results_per_platform)
            all_offers.extend(offers)
            print(f"✓ ({len(offers)})")
            time.sleep(0.5)

        # eBay
        if "ebay" in platforms:
            print("  🏪 eBay...", end=" ")
            offers = self.search_ebay(query, max_results_per_platform)
            all_offers.extend(offers)
            print(f"✓ ({len(offers)})")
            time.sleep(0.5)

        # AliExpress
        if "aliexpress" in platforms:
            print("  🇨🇳 AliExpress...", end=" ")
            offers = self.search_aliexpress(query, max_results_per_platform)
            all_offers.extend(offers)
            print(f"✓ ({len(offers)})")
            time.sleep(0.5)

        # Google (zawsze)
        if "google" in platforms:
            print("  🔎 Google...", end=" ")
            offers = self.search_google(query, max_results_per_platform)
            all_offers.extend(offers)
            print(f"✓ ({len(offers)})")

        return all_offers


def export_offers_to_html(
    offers: List[OnlineOffer],
    part_id: str,
    description: str,
    output_file: str = "oferty.html",
) -> str:
    """
    Eksportuj oferty do pliku HTML.
    
    Args:
        offers: Lista znalezionych ofert
        part_id: ID części
        description: Opis części
        output_file: Ścieżka do pliku wyjściowego
        
    Returns:
        Ścieżka do pliku HTML
    """
    html_content = f"""<!DOCTYPE html>
<html lang="pl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Oferty dla: {part_id}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 8px;
            margin-bottom: 30px;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
        }}
        .header p {{
            margin: 10px 0 0 0;
            opacity: 0.9;
        }}
        .offers-container {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }}
        .offer-card {{
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
            padding: 20px;
            transition: transform 0.2s, box-shadow 0.2s;
            display: flex;
            flex-direction: column;
        }}
        .offer-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        }}
        .offer-source {{
            display: inline-block;
            background: #667eea;
            color: white;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.8em;
            font-weight: bold;
            margin-bottom: 10px;
            width: fit-content;
        }}
        .offer-title {{
            font-size: 1.2em;
            font-weight: bold;
            margin: 10px 0;
            color: #333;
        }}
        .offer-seller {{
            color: #666;
            font-size: 0.9em;
            margin: 5px 0;
        }}
        .offer-price {{
            font-size: 1.5em;
            color: #27ae60;
            font-weight: bold;
            margin: 10px 0;
        }}
        .offer-availability {{
            color: #e74c3c;
            font-size: 0.9em;
            margin: 5px 0;
        }}
        .offer-description {{
            color: #555;
            font-size: 0.9em;
            margin: 10px 0;
            flex-grow: 1;
        }}
        .offer-link {{
            display: inline-block;
            background: #667eea;
            color: white;
            padding: 10px 20px;
            border-radius: 4px;
            text-decoration: none;
            margin-top: 15px;
            text-align: center;
            transition: background 0.2s;
        }}
        .offer-link:hover {{
            background: #764ba2;
        }}
        .no-offers {{
            background: white;
            padding: 40px;
            text-align: center;
            border-radius: 8px;
            color: #666;
        }}
        .footer {{
            text-align: center;
            color: #999;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔍 Oferty do zakupu</h1>
        <p><strong>Część:</strong> {part_id} - {description}</p>
        <p><strong>Wygenerowano:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="offers-container">
"""

    if not offers:
        html_content += """
        <div class="no-offers">
            <p>❌ Brak znalezionych ofert. Spróbuj wyszukać ręcznie na platformach e-commerce.</p>
        </div>
"""
    else:
        for offer in offers:
            html_content += f"""
        <div class="offer-card">
            <span class="offer-source">{offer.source.upper()}</span>
            <h3 class="offer-title">{offer.title}</h3>
            <p class="offer-seller">📌 {offer.seller}</p>
"""
            if offer.price:
                html_content += f'            <p class="offer-price">{offer.price} {offer.currency}</p>\n'

            if offer.availability != "unknown":
                html_content += f'            <p class="offer-availability">📦 {offer.availability}</p>\n'

            if offer.rating:
                html_content += f'            <p class="offer-availability">⭐ Ocena: {offer.rating}/5</p>\n'

            if offer.description:
                html_content += f'            <p class="offer-description">{offer.description}</p>\n'

            html_content += f'            <a href="{offer.url}" target="_blank" class="offer-link">Przejdź do oferty →</a>\n'
            html_content += """        </div>
"""

    html_content += """
    </div>
    
    <div class="footer">
        <p>💡 Kliknij na "Przejdź do oferty", aby przejść do platformy i zakupić część.</p>
        <p>📱 Strona wygenerowana automatycznie przez system RAG dla części metalowych.</p>
    </div>
    
    <div style="background: white; padding: 30px; border-radius: 8px; margin-top: 40px;">
        <h2 style="border-bottom: 2px solid #667eea; padding-bottom: 10px;">📋 Spis linków do ofert</h2>
"""
    
    if offers:
        for idx, offer in enumerate(offers, 1):
            html_content += f"""
        <div style="margin: 20px 0; padding: 15px; background: #f9f9f9; border-left: 4px solid #667eea;">
            <p style="margin: 0 0 10px 0;"><strong>{idx}. Opis:</strong> {offer.title}</p>
            <p style="margin: 0;"><strong>Link:</strong> <a href="{offer.url}" target="_blank" style="color: #667eea; text-decoration: none;">{offer.url}</a></p>
        </div>
"""
    else:
        html_content += "\n        <p style='color: #999;'>Brak ofert do wyświetlenia.</p>"
    
    html_content += """
    </div>
</body>
</html>
"""

    # Zapisz do pliku
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"\n✅ Oferty zapisane do: {output_file}")
    return output_file


def export_offers_to_json(
    offers: List[OnlineOffer],
    output_file: str = "oferty.json",
) -> str:
    """Eksportuj oferty do JSON."""
    # Utwórz listę linków w formacie: nr. Opis. Link.
    links_list = []
    for idx, offer in enumerate(offers, 1):
        links_list.append({
            "number": idx,
            "description": offer.title,
            "url": offer.url
        })
    
    data = {
        "timestamp": datetime.now().isoformat(),
        "total_offers": len(offers),
        "offers": [offer.to_dict() for offer in offers],
        "links": links_list,  # Dodaj spis linków
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"✅ Oferty zapisane do JSON: {output_file}")
    return output_file


def export_offers_to_csv(
    offers: List[OnlineOffer],
    output_file: str = "oferty.csv",
) -> str:
    """Eksportuj oferty do CSV."""
    try:
        import csv
    except ImportError:
        print("❌ Moduł csv niedostępny")
        return ""

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OnlineOffer.__dataclass_fields__.keys())
        writer.writeheader()
        for offer in offers:
            writer.writerow(offer.to_dict())

    print(f"✅ Oferty zapisane do CSV: {output_file}")
    return output_file


if __name__ == "__main__":
    # Test
    finder = OfferFinder()
    offers = finder.search_all_platforms(
        "śruba M8",
        platforms=["allegro", "amazon", "ebay", "google"],
        max_results_per_platform=2,
    )

    print(f"\n📊 Znaleziono {len(offers)} ofert\n")

    # Eksport
    export_offers_to_html(
        offers, "SCR-001", "Śruba M8 ze stali", output_file="/tmp/test_oferty.html"
    )
    export_offers_to_json(offers, output_file="/tmp/test_oferty.json")
