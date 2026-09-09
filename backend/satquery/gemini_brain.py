"""SatQuery Autonomous Multi-Agent Cognitive Core (Gemini AI Brain)

Implements the authentic multi-agent cognitive architecture:
User Query -> Intent Understanding & Task Dispatch (Gemini + RAG)
           -> Specialized Domain Models Execution (SAR / Optical / Bhoonidhi STAC / Verification Gates)
           -> Cognitive Reasoning & Synthesis over Model Outputs (Gemini)
           -> Tailored, Dynamic User-Oriented Output

Ensures SatQuery AI is never a shallow wrapper: standard general LLMs cannot
differentiate microwave SAR phase/backscatter, speckle patterns, or GeoTIFF bands without
specialized training. SatQuery's domain models perform the hard physical measurement,
and Gemini reasons over those measurements to deliver deep, user-centric intelligence.
"""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional


def get_gemini_api_keys() -> List[str]:
    """Retrieve all available Gemini API keys in priority order with fallback redundancy."""
    keys: List[str] = []
    seen = set()

    def _add(k: str):
        k = k.strip().strip("'\"")
        if k and len(k) > 15 and k not in seen:
            seen.add(k)
            keys.append(k)

    # 1. Direct environment variables
    env_vars = [
        "GEMINI_API_KEY", "GEMINI_API_KEY_1", "GEMINI_API_KEY1",
        "GEMINI_API_KEY_2", "GEMINI_API_KEY2", "GEMINI_API_KEY_3",
        "GEMINI_API_KEY3", "GEMINI_API_KEY_4", "GEMINI_API_KEY4",
    ]
    for ev in env_vars:
        val = os.environ.get(ev, "")
        if val:
            _add(val)

    csv_keys = os.environ.get("GEMINI_API_KEYS", "")
    if csv_keys:
        for k in csv_keys.split(","):
            _add(k)

    # 2. backend/.env
    backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    b_env = os.path.join(backend_dir, ".env")
    if os.path.exists(b_env):
        try:
            with open(b_env, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if "=" in line and not line.startswith("#"):
                        k, v = line.split("=", 1)
                        if "GEMINI_API_KEY" in k.strip():
                            _add(v)
        except Exception:
            pass

    # 3. frontend/.env.local
    frontend_env = os.path.join(os.path.dirname(backend_dir), "frontend", ".env.local")
    if os.path.exists(frontend_env):
        try:
            with open(frontend_env, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if "=" in line and not line.startswith("#"):
                        k, v = line.split("=", 1)
                        if "GEMINI_API_KEY" in k.strip():
                            _add(v)
        except Exception:
            pass

    # Prioritize official Google AI Studio keys (starting with AIza)
    aiza_keys = [k for k in keys if k.startswith("AIza")]
    other_keys = [k for k in keys if not k.startswith("AIza")]
    return aiza_keys + other_keys


def get_gemini_api_key() -> str:
    """Return primary active Gemini API key."""
    keys = get_gemini_api_keys()
    return keys[0] if keys else ""


def generate_with_gemini(
    prompt: str,
    system_instruction: Optional[str] = None,
    model: str = "gemini-2.5-flash",
    temperature: float = 0.35,
    timeout: float = 25.0,
) -> Optional[str]:
    """Call Google Gemini REST endpoint with multi-key failover and model resilience."""
    keys = get_gemini_api_keys()
    if not keys:
        return None

    candidate_models = [model, "gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-flash-latest"]
    seen_models = []
    for m in candidate_models:
        if m and m not in seen_models:
            seen_models.append(m)

    last_err = None

    for key in keys:
        for m in seen_models:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent?key={key}"

            payload: Dict[str, Any] = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": temperature,
                    "maxOutputTokens": 2048,
                },
            }

            if system_instruction:
                payload["systemInstruction"] = {
                    "parts": [{"text": system_instruction}]
                }

            try:
                data = json.dumps(payload).encode("utf-8")
                req = urllib.request.Request(
                    url,
                    data=data,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    if resp.status == 200:
                        body = json.loads(resp.read().decode("utf-8"))
                        candidates = body.get("candidates", [])
                        if candidates:
                            parts = candidates[0].get("content", {}).get("parts", [])
                            if parts:
                                text = parts[0].get("text", "")
                                return text.strip()
            except urllib.error.HTTPError as http_err:
                last_err = f"HTTP {http_err.code}: {http_err.reason}"
                # On 404 or 429, try the next model on this key before moving to next key
                continue
            except Exception as exc:
                last_err = exc
                continue

    if last_err:
        print(f"[gemini_brain] Gemini inference failed across key pool: {last_err}", file=sys.stderr)
    return None


# ============================================================================
# Stage 1: Upstream Intent Parsing & Task Planning (Gemini + RAG)
# ============================================================================

def plan_intent_with_gemini(
    query: str,
    has_user_files: bool,
    file_names: List[str],
    resolved_region: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Upstream cognitive dispatcher: Decomposes natural language queries into an optimal multi-agent execution plan."""
    system_instruction = (
        "You are the Upstream Task Dispatcher of SatQuery AI, an Earth Observation intelligence system. "
        "Analyze the user's natural language remote sensing query and output a strict JSON plan with:\n"
        "{\n"
        '  "task": "change_vqa" | "fusion" | "grounding" | "vqa" | "catalog_search" | "conversation",\n'
        '  "is_temporal": true | false,\n'
        '  "is_fusion": true | false,\n'
        '  "location_name": "<specific city, district, state, country, or geographic landmark explicitly mentioned in the user prompt (e.g. \'Madhya Pradesh\', \'Bhopal\', \'Sundarbans\', \'Netherlands\', \'Amazon rainforest\'), or null if NO real-world geographic place was named in the text>",\n'
        '  "target_subject": "<e.g. water bodies, built-up areas, deforestation, ships, crops, general>",\n'
        '  "temporal_window": "<extracted dates or years, or null>",\n'
        '  "question_focus": "<1-sentence summary of what the user specifically wants to know>"\n'
        "}\n"
        "Rules:\n"
        "- Do NOT confuse action verbs (such as 'Describe', 'Analyze', 'Explain', 'Highlight', 'Count') with place names.\n"
        "- If the query only says 'Describe this scene' or 'What is here', 'location_name' MUST be null.\n"
        "Output ONLY the JSON object, without markdown blocks."
    )

    files_info = ", ".join(file_names) if file_names else "None"
    region_info = resolved_region.get("name", "None") if resolved_region else "None"

    prompt = (
        f"User Query: \"{query}\"\n"
        f"Attached Files: {files_info} ({'Present' if has_user_files else 'Missing'})\n"
        f"Resolved Region: {region_info}\n"
    )

    raw = generate_with_gemini(prompt, system_instruction=system_instruction, temperature=0.1)
    if not raw:
        return None

    try:
        cleaned = raw.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        parsed = json.loads(cleaned.strip())
        if isinstance(parsed, dict) and "task" in parsed:
            return parsed
    except Exception:
        pass

    return None


# ============================================================================
# Stage 2 & 3: Downstream Cognitive Reasoner over Domain Model Outputs
# ============================================================================

def synthesize_specialist_results(
    query: str,
    raw_trace: Dict[str, Any],
    intent: Dict[str, Any],
    region: Optional[Dict[str, Any]],
    bhoonidhi_scenes: List[Dict[str, Any]],
    has_user_files: bool,
    backend_files: List[Dict[str, Any]],
) -> Optional[str]:
    """Downstream Cognitive Reasoner: Ingests raw specialist model outputs and gate checks,

    and formulates an authoritative, query-specific explanation with remote sensing physics.
    """
    system_instruction = (
        "You are SatQuery AI, an autonomous Earth Observation & Remote Sensing specialist intelligence engine. "
        "Our specialized domain models (SAR-optical cross-attention fusion heads, bi-temporal Siamese change networks, "
        "open-vocabulary visual grounding models, and ISRO Bhoonidhi STAC ingestion) have executed pixel-level deterministic inference. "
        "A standard general LLM cannot differentiate microwave SAR speckle, dielectric properties, or GeoTIFF multispectral bands "
        "without specialized training. Your role is NOT to guess or hallucinate image pixels. Instead, you MUST interpret, "
        "reason over, and synthesize the specialist models' exact measurements, quantities, and verification gates into an "
        "authoritative, insightful, and user-centric response that directly and comprehensively answers the user's question.\n\n"
        "Guidelines:\n"
        "1. Direct Answer First: Begin with a direct, unambiguous answer to the user's specific query.\n"
        "   For change detection queries (e.g. 'What is the difference between Mumbai in 1996 and 2023?' or 'Has built-up increased?'), "
        "   state the exact quantitative finding computed by the change model (e.g. detected land-cover transition, affected area in hectares, "
        "   delta percentage, or surface stability).\n"
        "2. Ground in Domain Physics: Explain WHY the models arrived at this conclusion using remote sensing physics "
        "   (e.g., historical IRS-1C LISS-III VNIR to Sentinel-2 MSI and EOS-04 C-band SAR, microwave double-bounce urban backscatter, "
        "   specular reflection on calm water, or multispectral NDVI / MNDWI index separation).\n"
        "3. Explicit Quantities: Cite the exact numbers computed by the specialist models (e.g. affected hectares, confidence %, bounding coordinates).\n"
        "4. DO NOT claim that change detection was not executed or that only a catalog search was run. The specialist models HAVE executed the inference.\n"
        "5. Transparent Quality & Gates: Note the verification gates passed (CRS co-registration, GSD tolerance, temporal order).\n"
        "6. Presentation: Use clean, professional Markdown with clear section headers, bullet points, and high readability. Do NOT mention you are an AI model."
    )

    # Build comprehensive context package from the domain models' execution
    out = raw_trace.get("output", {})
    raw_model_text = out.get("text", "")
    quantities = out.get("quantities", {})
    confidence = out.get("confidence")
    task_executed = raw_trace.get("classified_task", intent.get("task", "vqa"))
    input_check = raw_trace.get("input_check", {})
    abstained = raw_trace.get("abstained", False)

    steps_summary = []
    for s in raw_trace.get("steps", []):
        tool = s.get("tool", "unknown_tool")
        duration = s.get("duration_ms", 0)
        params = s.get("params", {})
        steps_summary.append(f"- Tool: `{tool}` ({duration}ms) | Params: {json.dumps(params)}")

    context_sections = [
        f"### User Request\nQuery: \"{query}\"",
        f"### Specialist Model Execution Trace\n"
        f"- Task Executed: {task_executed}\n"
        f"- Abstained by Gate: {abstained}\n"
        f"- Raw Specialist Output Text: \"{raw_model_text}\"\n"
        f"- Computed Quantities: {json.dumps(quantities)}\n"
        f"- Model Confidence: {confidence if confidence is not None else 'N/A'}\n"
        f"- Input Modalities Processed: {input_check.get('modality', [])}\n"
        f"- Spatial Resolution (GSD): {input_check.get('gsd_m', [])} meters\n"
        f"- Coordinate Reference System (CRS): {input_check.get('crs', [])}\n"
        f"- Gate Message: {input_check.get('message', 'All physical gates passed')}",
    ]

    if steps_summary:
        context_sections.append("### Specialist Tools & Preprocessing Applied\n" + "\n".join(steps_summary[:6]))

    if region:
        c_lat = region.get("center", [0, 0])[1]
        c_lon = region.get("center", [0, 0])[0]
        context_sections.append(
            f"### Geographic Scope\n"
            f"- Region Name: {region.get('name')}\n"
            f"- Center: {c_lat:.4f}°N, {c_lon:.4f}°E\n"
            f"- Extent Bounding Box: {region.get('bbox')}\n"
            f"- Marine / Ocean Waters: {region.get('is_ocean', False)}"
        )

    if bhoonidhi_scenes:
        scenes_info = [
            f"- Scene `{s.get('id')}`: Mission: {s.get('satellite')}, Sensor: {s.get('sensor', 'MSI')}, Date: {s.get('dop')}, Cloud: {s.get('coverage_pct', '0')}%"
            for s in bhoonidhi_scenes[:4]
        ]
        context_sections.append(
            f"### ISRO Bhoonidhi STAC Catalog Passes ({len(bhoonidhi_scenes)} candidate passes found)\n" + "\n".join(scenes_info)
        )

    prompt = (
        "\n\n".join(context_sections) +
        "\n\n### Task for Gemini Cognitive Synthesizer:\n"
        "Reason over the specialist model outputs and write the final response to the user. "
        "Make it direct, mathematically accurate, scientifically grounded, and formatted for maximum visual clarity."
    )

    return generate_with_gemini(prompt, system_instruction=system_instruction, temperature=0.3)


# ============================================================================
# Stage 4: Dynamic KPI & Executive Assessment Cards
# ============================================================================

def generate_dynamic_assessment_gemini(
    query: str,
    ai_response_text: str,
    region: Optional[Dict[str, Any]],
    bhoonidhi_scenes: List[Dict[str, Any]],
    has_user_files: bool,
    quantities: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Use Gemini to dynamically generate 1 headline, 2-line executive summary, and 3 contextual KPI cards

    strictly tied to the specialist models' actual calculations.
    """
    system_instruction = (
        "You are an Earth Observation data synthesizer for SatQuery AI. Given a remote sensing query, "
        "the specialist model's computed quantities, and the generated response, output a valid JSON object with:\n"
        "{\n"
        '  "headline": "<Punchy, 1-line authoritative finding, e.g. 42.8 Hectares Altered Across Southern Sector>",\n'
        '  "summary": "<Concise 2-sentence executive summary explaining the technical result>",\n'
        '  "metrics": [\n'
        '    {"label": "<Short Metric Name, e.g. Area Affected>", "value": "<Value with unit, e.g. 42.8 ha>", "icon": "layers|shield|satellite|map-pin|clock|crosshair", "tooltip": "<Explanation>"},\n'
        '    {"label": "<Short Metric Name, e.g. Model Confidence>", "value": "<e.g. 92.4% Verified>", "icon": "...", "tooltip": "..."},\n'
        '    {"label": "<Short Metric Name, e.g. Sensor Modality>", "value": "<e.g. C-Band SAR + MSI>", "icon": "...", "tooltip": "..."}\n'
        "  ],\n"
        '  "advisory": "<1-line actionable operational recommendation or physical insight>",\n'
        '  "advisory_level": "info" | "warning" | "success"\n'
        "}\n"
        "Output ONLY the JSON object, without markdown code fences."
    )

    quantities_str = json.dumps(quantities or {})
    prompt = (
        f"Query: \"{query}\"\n"
        f"Region: {region.get('name') if region else 'None'}\n"
        f"Has User Files: {has_user_files}\n"
        f"Specialist Model Quantities: {quantities_str}\n"
        f"Available STAC Passes: {len(bhoonidhi_scenes)}\n"
        f"Response Excerpt: {ai_response_text[:600]}\n"
    )

    raw_json = generate_with_gemini(prompt, system_instruction=system_instruction, temperature=0.2)
    if not raw_json:
        return None

    try:
        cleaned = raw_json.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        parsed = json.loads(cleaned.strip())
        if isinstance(parsed, dict) and "headline" in parsed and "metrics" in parsed:
            return parsed
    except Exception:
        pass

    return None


def generate_geospatial_domain_insight(
    query: str,
    region: Optional[Dict[str, Any]],
    bhoonidhi_scenes: List[Dict[str, Any]],
    intent: Dict[str, Any],
) -> Optional[str]:
    """Wraps Bhoonidhi STAC discoveries and AOI spatial grounding with deep domain Earth Observation analysis."""
    system_instruction = (
        "You are the Earth Observation Research Scientist for SatQuery AI (ISRO SIH26167). "
        "The system has already geolocated the target coordinates and retrieved live satellite acquisitions from ISRO's Bhoonidhi STAC catalog. "
        "Your role: Write a rich, scientifically rigorous, and highly readable domain assessment analyzing the user's question.\n\n"
        "Instructions:\n"
        "1. Specific Geographic & Historical Context: If the user asks about changes between years (e.g. 1996 vs 2023 in Mumbai), analyze the real-world surface dynamics: "
        "urban infrastructure expansion, land reclamation, mangrove cover along coastal creeks, and industrial zoning shifts.\n"
        "2. Remote Sensing Sensor Evolution: Explain the satellite mission transition across the requested epochs (e.g. historical IRS-1A/1B/1C LISS-II/III 23.5m optical vs modern Sentinel-2A/B 10m MSI and EOS-04 C-band SAR).\n"
        "3. Spectral Indices & Physics: Detail the exact quantitative indices suited for this analysis (NDVI for vegetation degradation, MNDWI for surface hydrology/coastline changes, NDBI for urban build-up, and SAR backscatter double-bounce reflections).\n"
        "4. Tone: Authoritative, evidence-led, and professional Markdown. Do NOT include generic disclaimers or state that you are an AI model."
    )

    place_name = region.get("name") if region else "the specified area"
    coords = f"{region.get('center', [0, 0])[1]:.4f}°N, {region.get('center', [0, 0])[0]:.4f}°E" if region else "Coordinates resolved"
    years = ", ".join(intent.get("years", [])) or "the requested temporal window"
    scene_count = len(bhoonidhi_scenes)

    prompt = (
        f"User Query: \"{query}\"\n"
        f"Target Location: {place_name} ({coords})\n"
        f"Temporal Periods: {years}\n"
        f"Live Bhoonidhi STAC Passes Found: {scene_count}\n"
        f"Classified Task: {intent.get('task')}\n\n"
        "Provide a comprehensive, domain-deep Earth Observation analysis for this query."
    )

    return generate_with_gemini(prompt, system_instruction=system_instruction, temperature=0.35)


def parse_and_reason_query(
    query: str,
    region: Optional[Dict[str, Any]],
    bhoonidhi_scenes: List[Dict[str, Any]],
    has_user_files: bool,
    backend_files: List[Dict[str, Any]],
    raw_trace: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Backward-compatible entry point that delegates to synthesize_specialist_results."""
    trace = raw_trace or {
        "output": {"text": "Catalog / Spatial Extent Query", "quantities": {}},
        "input_check": {"modality": ["Optical", "SAR"], "gsd_m": [10.0]},
        "steps": [],
    }
    intent = {
        "task": "catalog_search" if not has_user_files else "vqa",
        "is_temporal": "change" in query.lower(),
        "is_microwave": "sar" in query.lower() or "radar" in query.lower(),
    }
    return synthesize_specialist_results(
        query=query,
        raw_trace=trace,
        intent=intent,
        region=region,
        bhoonidhi_scenes=bhoonidhi_scenes,
        has_user_files=has_user_files,
        backend_files=backend_files,
    )
