import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("MVTEC_ROOT", PROJECT_ROOT / "data" / "mvtec_anomaly_detection"))

CACHE_DIR = PROJECT_ROOT / "cache"
SPLITS_DIR = PROJECT_ROOT / "splits"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"
TABLES_DIR = PROJECT_ROOT / "tables"

TEXTURES = ("carpet", "grid", "leather", "tile", "wood")
OBJECTS = (
    "bottle", "cable", "capsule", "hazelnut", "metal_nut",
    "pill", "screw", "toothbrush", "transistor", "zipper",
)
CATEGORIES = TEXTURES + OBJECTS

CATEGORY_NAMES_PL = {
    "carpet": "dywan",
    "grid": "siatka",
    "leather": "skóra",
    "tile": "płytka",
    "wood": "drewno",
    "bottle": "butelka",
    "cable": "kabel",
    "capsule": "kapsułka",
    "hazelnut": "orzech laskowy",
    "metal_nut": "nakrętka",
    "pill": "tabletka",
    "screw": "śruba",
    "toothbrush": "szczoteczka do zębów",
    "transistor": "tranzystor",
    "zipper": "zamek błyskawiczny",
}

DEFECT_NAMES_PL = {
    "good": "dobry",
    "bent": "wygięcie",
    "bent_lead": "wygięta nóżka",
    "bent_wire": "wygięty drut",
    "broken": "przerwanie",
    "broken_large": "duży ubytek",
    "broken_small": "mały ubytek",
    "broken_teeth": "złamane ząbki",
    "cable_swap": "zamienione przewody",
    "color": "przebarwienie",
    "combined": "defekty łączone",
    "contamination": "zanieczyszczenie",
    "crack": "pęknięcie",
    "cut": "nacięcie",
    "cut_inner_insulation": "przecięta izolacja wewnętrzna",
    "cut_lead": "ucięta nóżka",
    "cut_outer_insulation": "przecięta izolacja zewnętrzna",
    "damaged_case": "uszkodzona obudowa",
    "defective": "wadliwe włosie",
    "fabric_border": "uszkodzona krawędź taśmy",
    "fabric_interior": "uszkodzona taśma",
    "faulty_imprint": "wadliwy nadruk",
    "flip": "odwrócenie",
    "fold": "zagięcie",
    "glue": "plama kleju",
    "glue_strip": "pasek kleju",
    "gray_stroke": "szara smuga",
    "hole": "dziura",
    "liquid": "plama cieczy",
    "manipulated_front": "uszkodzony czubek",
    "metal_contamination": "zanieczyszczenie metalem",
    "misplaced": "nieprawidłowe położenie",
    "missing_cable": "brak przewodu",
    "missing_wire": "brak drutów",
    "oil": "plama oleju",
    "pill_type": "niewłaściwy typ tabletki",
    "poke": "przebicie",
    "poke_insulation": "przebita izolacja",
    "print": "nadruk",
    "rough": "chropowatość",
    "scratch": "zarysowanie",
    "scratch_head": "zarysowanie łba",
    "scratch_neck": "zarysowanie szyjki",
    "split_teeth": "rozdzielone ząbki",
    "squeeze": "zgniecenie",
    "squeezed_teeth": "zgniecione ząbki",
    "thread": "nitka",
    "thread_side": "uszkodzony gwint z boku",
    "thread_top": "uszkodzony gwint u góry",
}

RESIZE = 256
CROP = 224

SPLIT_SEED = 42
VAL_FRACTION = 0.30
HOLDOUT_FRACTION = 0.10
SPLIT_FILE = SPLITS_DIR / f"split_seed{SPLIT_SEED}.csv"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


CATEGORY_TYPE_NAMES_PL = {"texture": "tekstura", "object": "obiekt"}


def category_type(category: str) -> str:
    if category in TEXTURES:
        return "texture"
    if category in OBJECTS:
        return "object"
    raise ValueError(f"unknown category: {category}")


def category_type_name_pl(category_type_key: str) -> str:
    return CATEGORY_TYPE_NAMES_PL[category_type_key]


def category_name_pl(category: str) -> str:
    return CATEGORY_NAMES_PL[category]


def defect_name_pl(defect_type: str) -> str:
    return DEFECT_NAMES_PL[defect_type]
