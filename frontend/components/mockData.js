/**
 * SatQuery AI (ISRO SIH26167)
 * Authentic domain templates and folder taxonomy.
 * Dead starter code removed.
 */

export const INITIAL_FOLDERS = [
  { id: "f1", name: "ISRO SIH26167 Missions" },
  { id: "f2", name: "Optical & Multispectral Analysis" },
  { id: "f3", name: "SAR & Microwave Grounding" },
  { id: "f4", name: "Bi-Temporal Change Detection" },
]

export const INITIAL_TEMPLATES = [
  {
    id: "t1",
    name: "Joint Optical + SAR Fusion (Rule R1)",
    content: `**Joint Optical-SAR Fusion Analysis**

**Target Geographic Region:**
[Specify coordinates or location name, e.g. Sundarbans or Delhi]

**Sensors & Modalities:**
- Optical/MSI: Sentinel-2A (10m GSD) / Cartosat-2S
- Radar/SAR: EOS-04 C-band SAR (VV/VH dual-pol)

**Analysis Query:**
Use the optical and SAR images together to identify built-up structures and water-covered regions. Quantify areas in hectares and highlight corner reflectors.`,
    snippet: "Cross-modal optical and SAR joint land-cover analysis template...",
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
  },
  {
    id: "t2",
    name: "Bi-Temporal Change Detection (Rule R3)",
    content: `**Bi-Temporal Change Detection Query**

**Baseline Observation (T1):** [e.g. 2024-11-12]
**Current Observation (T2):** [e.g. 2025-11-15]

**Target Feature:**
[Built-up infrastructure / Surface Water / Low Vegetation]

**Analysis Query:**
Did the built-up area increase between the two observation dates? Delineate newly constructed footprints and report percentage delta.`,
    snippet: "Multi-temporal pair change detection and urban growth quantification...",
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
  },
  {
    id: "t3",
    name: "Spatial Grounding & Localization (Rule R4)",
    content: `**Text-Guided Spatial Grounding**

**Target Class:**
[e.g. Water bodies, pastures, industrial facility, airport runway]

**Analysis Query:**
Where is the largest connected region of pastures? Highlight the bounding box coordinates [ymin, xmin, ymax, xmax] in normalized coordinates.`,
    snippet: "Spatial bounding box extraction using fine-tuned LoRA grounding adapter...",
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
  },
  {
    id: "t4",
    name: "High-Resolution Scene VQA (Rule R6)",
    content: `**Single-Image Visual Question Answering**

**Target Image:** [lr_232.tif or opt.tif]

**Analysis Query:**
Is there a road visible in the scene? What is the settlement density (rural vs urban)? Count distinct building footprints if resolvable at current GSD.`,
    snippet: "Specialist VLM question-answering on high-resolution satellite imagery...",
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
  },
]
