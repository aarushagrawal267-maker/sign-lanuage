"""Spelling lexicon: vocabulary headwords + Indian states, union territories and cities (with Madhya Pradesh
districts, since the first users are in Indore). Log priors: places and core words slightly above other words.
SCOWL (a permissive English word list) can be appended later as ``word<TAB>log_prior`` lines."""

from __future__ import annotations

import math
import re

PLACES = """
Andhra Pradesh, Arunachal Pradesh, Assam, Bihar, Chhattisgarh, Goa, Gujarat, Haryana, Himachal Pradesh, Jharkhand,
Karnataka, Kerala, Madhya Pradesh, Maharashtra, Manipur, Meghalaya, Mizoram, Nagaland, Odisha, Punjab, Rajasthan,
Sikkim, Tamil Nadu, Telangana, Tripura, Uttar Pradesh, Uttarakhand, West Bengal, Delhi, Chandigarh, Puducherry,
Ladakh, Jammu, Kashmir, Lakshadweep, Andaman, Nicobar, India, Bharat,
Mumbai, Kolkata, Chennai, Bengaluru, Bangalore, Hyderabad, Ahmedabad, Pune, Surat, Jaipur, Lucknow, Kanpur, Nagpur,
Patna, Vadodara, Ludhiana, Agra, Nashik, Varanasi, Srinagar, Amritsar, Prayagraj, Allahabad, Ranchi, Coimbatore,
Madurai, Kochi, Thiruvananthapuram, Visakhapatnam, Vijayawada, Mysuru, Mysore, Guwahati, Bhubaneswar, Dehradun,
Shimla, Raipur, Noida, Gurugram, Gurgaon, Faridabad, Ghaziabad, Meerut, Rajkot, Jodhpur, Udaipur, Kota, Ajmer,
Indore, Bhopal, Jabalpur, Gwalior, Ujjain, Sagar, Dewas, Satna, Ratlam, Rewa, Katni, Singrauli, Burhanpur, Khandwa,
Khargone, Dhar, Jhabua, Alirajpur, Barwani, Mandsaur, Neemuch, Shajapur, Agar, Rajgarh, Vidisha, Raisen, Sehore,
Hoshangabad, Narmadapuram, Betul, Harda, Chhindwara, Seoni, Balaghat, Mandla, Dindori, Shahdol, Umaria, Anuppur,
Sidhi, Damoh, Panna, Chhatarpur, Tikamgarh, Niwari, Datia, Shivpuri, Guna, Ashoknagar, Morena, Bhind, Sheopur,
Narsinghpur, Mhow, Pithampur, Omkareshwar, Maheshwar, Mandu, Khajuraho, Pachmarhi, Sanchi, Orchha, Narmada
"""


def place_names() -> list[str]:
    return [p.strip() for p in PLACES.replace("\n", " ").split(",") if p.strip()]


def build_spelling_lexicon(vocab_rows: list[dict]) -> list[tuple[str, float]]:
    out: dict[str, float] = {}
    for r in vocab_rows:
        kind = r.get("kind", "word")
        hw = (r.get("headword") or "").strip()
        if kind not in ("word", "number", "multiword") or not hw or len(hw) > 30:
            continue
        if not re.fullmatch(r"[A-Za-z][A-Za-z '\-]*", hw):
            continue
        lp = math.log(0.5) if r.get("in_classifier") else math.log(0.2)
        if kind == "multiword":
            lp += math.log(0.5)
        out[hw] = max(out.get(hw, -1e9), lp)
    for p in place_names():
        out[p] = max(out.get(p, -1e9), math.log(0.6))
    return sorted(out.items())


def write_spelling_lexicon(rows: list[tuple[str, float]], path) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# word<TAB>log prior (vocabulary headwords + Indian place names)\n")
        for w, lp in rows:
            fh.write(f"{w}\t{lp:.4f}\n")
