# KRISHI – Knowledge-driven Real-time Intelligent System for Harvest Improvement

Production-style, hackathon-friendly MVP: login, image upload, OpenCV-based vegetation/stress analysis, trust score, and advisory.

---

## Hackathon-winning features

- **Demo mode** — In Image Analyzer, click **Try with sample image** to run a full analysis without uploading a file. Perfect for live demos and judges.
- **One-line Judge summary** — After analysis, the Dashboard shows *Your latest plot at a glance*: suitability and top crop in one sentence.
- **Download / Print report** — After analysis or AI Suggestion, use **Download / Print report** to open a one-page summary and save as PDF or print.
- **AI Suggestion tab** — Select a plot (last analyzed or current location) and get suitability + crop suggestions in one go.
- **Best masking** — Optional Hugging Face token for semantic land-use (roads, buildings, water, greenery, forest, clear land). See README section on masking.

---

## How to open (complete steps)

1. **Install once** (if not done):
   ```powershell
   cd "c:\Users\Tony Stark\Desktop\projects\agni"
   pip install -r requirements.txt
   ```

2. **Start the backend** (leave this window open):
   ```powershell
   cd "c:\Users\Tony Stark\Desktop\projects\agni\backend"
   
   ```
   Wait until you see *Application startup complete*.

3. **Start the frontend** (open a **new** PowerShell window):
   ```powershell
   cd "c:\Users\Tony Stark\Desktop\projects\agni\frontend"
   python -m http.server 8080
   ```

4. **Open in browser:**  
   Go to **http://localhost:8080**

5. **Log in:**  
   Username: `krishi` · Password: `farm2025`  
   Then upload a field/crop image and click **Analyze image**.

---

## Quick start (reference)

### 1. Install dependencies

**PowerShell (path has spaces – use quotes):**

```powershell
cd "c:\Users\Tony Stark\Desktop\projects\agni"
pip install -r requirements.txt
```

### 2. Run the backend

**Option A – Run script (easiest in PowerShell):**

```powershell
cd "c:\Users\Tony Stark\Desktop\projects\agni\backend"
.\run.ps1
```

**Option B – Manual (quote the path, then run uvicorn):**

```powershell
cd "c:\Users\Tony Stark\Desktop\projects\agni\backend"
python -m uvicorn main:app --reload
```

Do not pass the path as an argument to uvicorn; only run the two commands above in order.

Backend runs at **http://127.0.0.1:8000**.  
Docs: http://127.0.0.1:8000/docs

### 3. Open the frontend

**Option A – Simple HTTP server (recommended)**

```powershell
cd "c:\Users\Tony Stark\Desktop\projects\agni\frontend"
python -m http.server 8080
```

Then open **http://localhost:8080** in your browser.

**Option B – Open file directly**

Open `frontend/index.html` in your browser (double-click or drag into Chrome/Edge).  
If the backend is at `http://127.0.0.1:8000`, login and analyze should still work thanks to CORS.

### 4. Test login and analyze

- **Login**  
  - Username: `krishi`  
  - Password: `farm2025`  
  - In the UI: enter credentials and click “Log in”.  
  - Or with curl:
    ```bash
    curl -X POST http://127.0.0.1:8000/login -H "Content-Type: application/json" -d "{\"username\":\"krishi\",\"password\":\"farm2025\"}"
    ```
  - Copy the `access_token` from the response.

- **Analyze**  
  - In the UI: after login, choose an image and click “Analyze image”.  
  - Or with curl (replace `YOUR_TOKEN` and `path/to/image.jpg`):
    ```bash
    curl -X POST http://127.0.0.1:8000/analyze -H "Authorization: Bearer YOUR_TOKEN" -F "file=@path/to/image.jpg"
    ```

## Best masking (what judges care about) – free AI, simple hardware

For **accurate semantic masking** (roads, buildings, clear land, greenery, forest, water), the app can use **AI semantic segmentation** via the free Hugging Face Inference API. One token enables it; no GPU or heavy hardware needed.

### How to enable

1. **Get a free token:** Sign up at [Hugging Face](https://huggingface.co/join) and create a token at [Settings → Access Tokens](https://huggingface.co/settings/tokens). Free tier is enough.
2. **Set the token** before starting the backend (PowerShell):
   ```powershell
   $env:HF_TOKEN = "hf_xxxxxxxx"
   cd backend
   python -m uvicorn main:app --reload
   ```
   Or set `BUILDING_SEGMENTATION_HF_TOKEN` instead of `HF_TOKEN` (same effect).
3. **Analyze an image** as usual. If the token is set, the backend calls the **SegFormer ADE20k** model (150 classes) once per image and produces:
   - **Land-use mask** with 6 classes: **road** (gray), **building** (brown), **water** (blue), **clear land** (tan), **greenery** (green), **forest** (dark green).
   - **Percentages** for each class in the API response (`water_percentage`, `road_percentage`, `building_percentage`, `clear_land_percentage`, `greenepython -m uvicorn main:app --reloadry_percentage`, `forest_percentage` when `semantic_masking_used` is `true`).

### When no token is set

The app falls back to **rule-based** masking (OpenCV: color, edges, contours). You still get water, roads, buildings, vegetation, bare soil, and stress; boundaries are smoothed for a cleaner land-use map.

### Smoother boundaries (less blocky/jagged)

- **AI path:** Masks are upscaled with linear interpolation and re-thresholded, then smoothed (Gaussian blur + light morphology) so edges look cleaner.
- **Optional higher resolution:** For even smoother results, set `HF_ADE20K_640=1` before starting the backend to use the 640×640 SegFormer model (one call per image; may be slower on free tier).

### Summary

| Feature | Free? | Hardware | What you get |
|--------|--------|----------|----------------|
| **AI masking (HF token)** | Yes (free tier) | Any (API runs in cloud) | Roads, buildings, water, clear land, greenery, forest |
| **Rule-based (no token)** | Yes | Any | Water, roads, buildings, vegetation, bare soil (color/shape only) |

No extra APIs to build; the backend uses the public Hugging Face Inference API. For best results in front of judges, set `HF_TOKEN` and use AI masking.

---

## Optional: Better building vs soil detection (rule-based + HF Cityscapes)

When **not** using the ADE20k semantic path above, buildings and bare soil are distinguished by:

1. **Rule-based:** Brown regions are classified as buildings only if compact and not mostly soil (low overlap with bare-soil mask and reasonably rectangular shape).
2. **Optional AI (same HF token):** If `HF_TOKEN` (or `BUILDING_SEGMENTATION_HF_TOKEN`) is set and ADE20k is not used for the image, the backend can call a Cityscapes segmentation model to add building pixels. No token = rule-based only (no API calls).

## Crop recommendations (real data)

Crop suggestions are **real**, not mock: a rule-based engine uses agronomic data (temperature ranges, humidity, water needs, seasons, space) for 50+ crops (cereals, pulses, vegetables, fruits, cash crops, spices, oilseeds). Season is derived from **latitude + month** (e.g. Kharif/Rabi/Zaid for India, spring/summer/fall/winter for temperate).

### Indicators used (better results with more inputs)

| Indicator | Required? | Description |
|-----------|-----------|-------------|
| **Temperature** | Yes | Current temp (°C) – from weather or manual |
| **Humidity** | Yes | Humidity % – from weather or manual |
| **Latitude** | Yes | For season and day length |
| **Cultivated %** | No (default 50) | Already cultivated (from image analysis) |
| **Water %** | No (default 5) | Water detected in image |
| **Farmable space %** | No (default 50) | Land available for farming |
| **Soil moisture** | No | `low` / `medium` / `high` or 0–100 – major indicator for crop fit |
| **Soil type** | No | e.g. `clay`, `loam`, `sandy`, `black alluvial`, `red laterite` – nature of soil in the region |
| **Soil pH** | No | `acidic` / `neutral` / `alkaline` |
| **Rainfall expected** | No | `low` / `medium` / `high` for the season |

Adding **soil moisture**, **soil type**, and **soil pH** (and optionally **rainfall expected**) improves ranking: crops are scored for soil and rainfall match and the response includes `indicators_used` and the values you sent.

**Example (with soil):**
```text
GET /recommend-crops?temp=28&humidity=65&lat=20&soil_moisture=high&soil_type=loam&soil_ph=neutral&rainfall_expected=high
```
(Add `Authorization: Bearer YOUR_TOKEN`.)

## Error handling

- **Invalid image** → 400 with message (e.g. “Invalid image: could not decode”).
- **Missing token** on `/analyze` → 401 “Missing token”.
- **Invalid or expired token** → 401 “Invalid or expired token”.
- All errors are JSON, e.g. `{"detail": "..."}`.

## Tech stack

- **Backend:** Python, FastAPI, OpenCV, NumPy, Uvicorn, JWT (python-jose).
- **Frontend:** HTML + Tailwind CSS (CDN), Fetch API.
- **Storage:** None (in-memory; mock user only).
