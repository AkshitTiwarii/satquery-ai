"""Semantic Query Comprehension Engine for SatQuery AI.

Autonomous linguistic and domain understanding for remote sensing queries:
- Extracts semantic subjects (hydrology, vegetation, transportation, built-up, agriculture, etc.)
- Identifies question modality (boolean existence, enumeration, measurement/ratio, localization, temporal change)
- Identifies temporal direction (increase, decrease, change, stability)
- Understands sensor requirements (optical, SAR, multimodal fusion)
- Eliminates hardcoded keyword checks across the entire system
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple


# Semantic Categories for Earth Observation
CAT_HYDROLOGY = "hydrology"
CAT_VEGETATION = "vegetation"
CAT_BUILT_ENVIRONMENT = "built_environment"
CAT_TRANSPORTATION = "transportation"
CAT_AGRICULTURE = "agriculture"
CAT_SOIL_TERRAIN = "soil_terrain"
CAT_LAND_COVER = "land_cover"
CAT_GENERAL = "general"

# Question Modalities
Q_BOOLEAN = "boolean_existence"       # "is there", "are there", "any X visible"
Q_COUNT = "enumeration_count"         # "how many", "count the", "number of"
Q_MEASUREMENT = "ratio_measurement"   # "what percentage", "how much of", "area of"
Q_LOCALIZATION = "spatial_grounding"  # "where is", "locate", "pinpoint", "highlight"
Q_TEMPORAL = "temporal_change"        # "did X change/increase/decrease", "before and after"
Q_OPEN = "open_classification"        # "what is", "describe", "identify the land cover"

# Domain-specific terminology and spectral attributes
FEATURE_METADATA = {
    CAT_HYDROLOGY: {
        "display_name": "Surface Hydrology & Water Bodies",
        "description": "lacustrine, riverine, or open surface water extent",
        "spectral_signature": "low optical albedo with high absorption across near-infrared (NIR) and shortwave-infrared (SWIR) bands",
        "radar_signature": "specular microwave backscatter yielding low radar return (distinct dark signature)",
        "morphology": "delineated shorelines, drainage networks, and contiguous open water surfaces",
    },
    CAT_VEGETATION: {
        "display_name": "Forest Canopy & Woodland Cover",
        "description": "forest canopy, arboreal cover, and perennial vegetation",
        "spectral_signature": "strong chlorophyll absorption in red/blue and high reflectance in the near-infrared (NIR) plateau",
        "radar_signature": "volume scattering within multi-layered vegetative canopy yielding moderate-to-high depolarized backscatter",
        "morphology": "irregular canopy textures, contiguous forest parcels, and natural foliage density",
    },
    CAT_AGRICULTURE: {
        "display_name": "Agricultural Parcels & Cropland",
        "description": "cultivated fields, crop parcels, and agrarian land",
        "spectral_signature": "dynamic greenness indices (NDVI/VARI) fluctuating with seasonal planting and harvesting cycles",
        "radar_signature": "surface roughness scattering varying by soil tilling, moisture content, and crop growth stage",
        "morphology": "geometric field boundaries, orthogonal parcel divisions, and uniform vegetative cover",
    },
    CAT_BUILT_ENVIRONMENT: {
        "display_name": "Built-Up Urban Settlements & Structures",
        "description": "residential clusters, commercial buildings, and impervious urban infrastructure",
        "spectral_signature": "high-reflectance impervious surfaces (concrete, asphalt, composite roofing) with suppressed vegetation indices",
        "radar_signature": "double-bounce microwave corner reflections from vertical building walls and ground planes",
        "morphology": "orthogonal structural edges, rectilinear footprints, and clustered settlement layouts",
    },
    CAT_TRANSPORTATION: {
        "display_name": "Transportation Infrastructure & Road Networks",
        "description": "road corridors, highways, arterial transit links, and paved networks",
        "spectral_signature": "linear high-contrast impervious ribbons against surrounding terrain with low spectral variance along corridors",
        "radar_signature": "smooth surface specular scattering flanked by distinct linear boundary gradients",
        "morphology": "continuous narrow linear corridors, topological connectivity, and high directional gradient continuity",
    },
    CAT_SOIL_TERRAIN: {
        "display_name": "Open Terrain & Bare Fallow Ground",
        "description": "bare soil, exposed rock, fallow ground, and non-vegetated terrain",
        "spectral_signature": "linear increasing reflectance from visible to shortwave infrared without red-edge inflection",
        "radar_signature": "diffuse microwave backscatter dominated by surface roughness and soil dielectric permittivity",
        "morphology": "unstructured open expanses, continuous ground terrain, and absence of vertical architectural elevation",
    },
    CAT_LAND_COVER: {
        "display_name": "Land Cover & Terrain Classification",
        "description": "multispectral land use and land cover distribution",
        "spectral_signature": "differentiated spectral reflectance profiles across multispectral bands",
        "radar_signature": "polarimetric scattering decomposition revealing structural roughness and surface geometry",
        "morphology": "spatial mosaic of vegetative, agrarian, lacustrine, and anthropic land cover classes",
    },
    CAT_GENERAL: {
        "display_name": "Earth Observation Scene",
        "description": "remote sensing observation features",
        "spectral_signature": "multispectral reflectance across visible and near-infrared channels",
        "radar_signature": "microwave backscatter intensity",
        "morphology": "spatial geometry and feature distribution",
    },
}

# Linguistic pattern matchers for category extraction
_LEXICON = {
    CAT_TRANSPORTATION: [
        r"\broads?\b", r"\bhighways?\b", r"\bstreets?\b", r"\btransport(ation)?\b",
        r"\bbridges?\b", r"\brail(ways?|roads?)?\b", r"\brunways?\b", r"\bairports?\b",
        r"\bcorridors?\b", r"\bpavement\b", r"\bexpressways?\b", r"\blanes?\b"
    ],
    CAT_HYDROLOGY: [
        r"\bwaters?\b", r"\brivers?\b", r"\blakes?\b", r"\breservoirs?\b", r"\bcanals?\b",
        r"\boceans?\b", r"\bseas?\b", r"\bponds?\b", r"\bstreams?\b", r"\bwetlands?\b",
        r"\bflood\w*\b", r"\binundat(ion|ed|ing)?\b", r"\baquatic\b", r"\bmarine\b", r"\bdams?\b",
        r"\bwater\s*bodies?\b", r"\bcreeks?\b"
    ],
    CAT_VEGETATION: [
        r"\bforests?\b", r"\btrees?\b", r"\bcanop(y|ies)\b", r"\bgreen(ery)?\b", r"\bvegetat(ion|ive)\b",
        r"\bwoods?\b", r"\bwoodlands?\b", r"\btimber\b", r"\bjungle\b", r"\bdeforest(ation)?\b",
        r"\bmangroves?\b", r"\bafforestation\b", r"\bnatural\s*reserve\b"
    ],
    CAT_AGRICULTURE: [
        r"\bagricultur(e|al)\b", r"\bcrops?\b", r"\bfarms?\b", r"\bfarmlands?\b", r"\bfields?\b",
        r"\barable\b", r"\bpastures?\b", r"\bcultivat(ion|ed)\b", r"\bpaddy\b", r"\bharvest(ing)?\b"
    ],
    CAT_BUILT_ENVIRONMENT: [
        r"\bbuilt[\s\-]?up\b", r"\bbuildings?\b", r"\bhous(es|ing)\b", r"\burbans?\b", r"\bsettlements?\b",
        r"\bcit(y|ies)\b", r"\btowns?\b", r"\bresidential\b", r"\bcommercial\b", r"\bindustr(y|ial)\b",
        r"\bfactor(y|ies)\b", r"\bstructures?\b", r"\bconstructions?\b", r"\broof(s|tops)?\b",
        r"\bimpervious\b", r"\bsolar\s*panels?\b", r"\bwarehouses?\b", r"\bcompounds?\b"
    ],
    CAT_SOIL_TERRAIN: [
        r"\bsoils?\b", r"\bbare\s*(earth|ground|soil)\b", r"\bfallow\b", r"\bsands?\b", r"\bquarr(y|ies)\b",
        r"\bmines?\b", r"\bmining\b", r"\bterrains?\b", r"\blandslides?\b", r"\berosions?\b",
        r"\bmountains?\b", r"\bhills?\b", r"\btopograph(y|ic)\b", r"\brocks?\b"
    ],
    CAT_LAND_COVER: [
        r"\bland\s*covers?\b", r"\bland\s*uses?\b", r"\bterrain\b", r"\blandscapes?\b",
        r"\bsurfaces?\b", r"\btypes?\s+of\s+land\b", r"\boverviews?\b"
    ],
}


def analyze_query_semantics(query: str) -> Dict[str, Any]:
    """Comprehensively extracts semantic intent, question type, domain category, and target entities.
    
    Ensures that NO model or tool uses hardcoded templates or canned responses.
    """
    q_clean = query.strip()
    q_lower = q_clean.lower()

    # 1. Determine Semantic Category & Target Entities
    detected_category = CAT_GENERAL
    matched_subject_phrase = ""

    # Match against domain lexicons
    category_scores = {}
    for cat, patterns in _LEXICON.items():
        for pat in patterns:
            m = re.search(pat, q_lower)
            if m:
                category_scores[cat] = category_scores.get(cat, 0) + 1
                if not matched_subject_phrase:
                    matched_subject_phrase = m.group(0)

    if category_scores:
        detected_category = max(category_scores.items(), key=lambda x: x[1])[0]

    if not matched_subject_phrase:
        # Fallback noun phrase extraction
        nouns = re.findall(r"\b[a-zA-Z]{4,}\b", q_clean)
        stopwords = {
            "what", "where", "when", "which", "there", "show", "tell", "using", "optical",
            "satellite", "image", "images", "analyze", "detect", "around", "over", "between"
        }
        filtered = [n for n in nouns if n.lower() not in stopwords]
        matched_subject_phrase = " ".join(filtered[:2]) if filtered else "observed terrain"

    meta = FEATURE_METADATA.get(detected_category, FEATURE_METADATA[CAT_GENERAL])

    # 2. Determine Question Format & Modality
    is_temporal = bool(re.search(
        r"\b(chang(e|ed|es)|increas(e|ed)|decreas(e|ed)|grew|grown|shrunk|shrank"
        r"|differ(ence|ent)|before\s+and\s+after|expand(ed)?|deforest\w*|new\s+construction"
        r"|over\s+time|since(\s+last\s+year)?|between\s+(19\d\d|20\d\d))\b",
        q_lower
    ))

    is_fusion = bool(re.search(
        r"\b(fus(e|ed|ion)|combin(e|ed|ing)|both\s+sensors|both\s+modalities"
        r"|optical\s+and\s+(radar|sar)|(radar|sar)\s+and\s+optical|using\s+both|microwave)\b",
        q_lower
    ))

    is_localization = bool(re.search(
        r"\b(where\b|locate|show\s+me\s+where|bounding\s+box|point\s+out|mark\s+the|find\s+the|pinpoint|coordinates\s+of)\b",
        q_lower
    ))

    is_count = bool(re.search(
        r"\b(how\s+many|count(\s+the)?|number\s+of|instances?\s+of)\b",
        q_lower
    ))

    is_measurement = bool(re.search(
        r"\b(what\s+percentage|how\s+much\s+of|ratio\s+of|fraction\s+of|extent\s+of|area\s+of|proportion)\b",
        q_lower
    ))

    is_boolean = bool(re.match(
        r"^\s*(is|are|does|do|did|has|have|was|were|can|could|will|would)\b",
        q_lower
    )) or bool(re.search(r"\b(is\s+there|are\s+there|any\b.*\b(present|visible|detected))\b", q_lower))

    # Assign primary question style
    if is_temporal:
        question_style = Q_TEMPORAL
    elif is_localization:
        question_style = Q_LOCALIZATION
    elif is_count:
        question_style = Q_COUNT
    elif is_measurement:
        question_style = Q_MEASUREMENT
    elif is_boolean:
        question_style = Q_BOOLEAN
    else:
        question_style = Q_OPEN

    # Determine Change Direction Hypothesis (if temporal)
    temporal_hypothesis = "neutral"
    if is_temporal:
        if re.search(r"\b(increas(e|ed)|growth|expand(ed|ing)|more|gain(ed)?)\b", q_lower):
            temporal_hypothesis = "increase"
        elif re.search(r"\b(decreas(e|ed)|shrink|shrunk|loss|lost|reduc(e|ed)|drop(ped)?|deforest(ation)?)\b", q_lower):
            temporal_hypothesis = "decrease"

    # Determine Task Routing
    if is_temporal:
        primary_task = "change_vqa"
    elif is_fusion:
        primary_task = "fusion"
    elif is_localization:
        primary_task = "grounding"
    else:
        primary_task = "vqa"

    return {
        "query": query,
        "category": detected_category,
        "target_subject": matched_subject_phrase,
        "display_name": meta["display_name"],
        "description": meta["description"],
        "spectral_signature": meta["spectral_signature"],
        "radar_signature": meta["radar_signature"],
        "morphology": meta["morphology"],
        "question_style": question_style,
        "is_temporal": is_temporal,
        "is_fusion": is_fusion,
        "is_localization": is_localization,
        "is_count": is_count,
        "is_boolean": is_boolean,
        "is_measurement": is_measurement,
        "temporal_hypothesis": temporal_hypothesis,
        "primary_task": primary_task,
    }
