"""
KRISHI – Knowledge-driven Real-time Intelligent System for Harvest Improvement.
FastAPI backend: auth, real OpenCV image analysis, trust score, advisory, crop recommendations.
"""
import base64
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Tuple

import cv2
import numpy as np
import requests
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Header, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.auth import verify_user, create_access_token, verify_token
from backend.image_processor import (
    analyze_image,
    get_segmentation_preview,
    get_overlay_mask,
    get_binary_mask_png,
    get_overlay_mask_png,
    get_landuse_mask_png,
    get_polygons_from_cultivated_mask,
    overlay_polygons_on_image,
    compute_deterministic_confidence,
    clip_cultivated_mask_by_boundary,
    get_semantic_masks_ade20k,
)
from backend.trust_engine import get_advisory
from backend.crop_database import recommend_crops, get_season, get_season_display_name

# Directory for saving cultivated overlay images (relative to backend)
OUTPUTS_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

app = FastAPI(title="KRISHI API", version="1.0.0")

# Serve saved overlay images at /outputs
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")


@app.get("/demo-image", response_class=Response)
def get_demo_image():
    """
    Returns a sample field image for demo/demo mode (hackathon: try without uploading).
    No auth required. Frontend fetches this and sends to /analyze for instant demo.
    """
    # Generate a simple 400x300 "field" image: green (vegetation) + brown (soil) + small gray (path)
    h, w = 300, 400
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :] = (34, 139, 34)  # green
    img[100:200, 50:350] = (139, 90, 43)   # brown soil patch
    img[140:160, 0:w] = (128, 128, 128)    # thin path
    img[0:40, 0:w] = (70, 130, 180)        # sky-like strip
    _, buf = cv2.imencode(".jpg", img)
    return Response(content=buf.tobytes(), media_type="image/jpeg")

# In-memory analysis history (last N entries used by GET /history)
analysis_history: List[dict] = []

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Request/Response models ---
class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


def _cultivation_classification(cultivated_percentage: float) -> str:
    """Classification from cultivated land share."""
    if cultivated_percentage > 70:
        return "Highly Cultivated"
    if cultivated_percentage >= 40:
        return "Moderately Cultivated"
    return "Poorly Cultivated"


class AnalyzeResponse(BaseModel):
    cultivated_percentage: float
    stress_percentage: float
    trust_score: float
    mask_confidence: float
    advisory: str
    is_blurry: bool
    classification: str
    total_pixels: int
    cultivated_pixels: int
    non_cultivated_pixels: int
    coverage_ratio: float
    contour_count: int
    sharpness_score: float
    veg_balance_score: float
    fragmentation_score: float
    noise_score: float
    laplacian_variance: float
    vegetation_ratio: float
    noise_ratio: float
    clarity_score: float
    edge_consistency_score: float
    area_plausibility_score: float
    exg_variance: float
    original_resolution: Tuple[int, int]
    processing_resolution: Tuple[int, int]
    segmentation_image_base64: Optional[str] = None
    binary_mask_base64: Optional[str] = None
    overlay_mask_base64: Optional[str] = None
    polygon_coordinates: Optional[List[List[List[float]]]] = None
    cultivated_mask_image_path: Optional[str] = None
    within_boundary: bool = False
    land_use_mask_base64: Optional[str] = None
    # When boundary is used: full image vs selected region vs remaining (outside)
    full_image_pixels: Optional[int] = None
    pixels_inside_boundary: Optional[int] = None
    pixels_outside_boundary: Optional[int] = None
    percentage_inside_boundary: Optional[float] = None
    percentage_outside_boundary: Optional[float] = None
    # Land-use breakdown percentages (rule-based or from AI semantic masking)
    water_percentage: float = 0.0
    road_percentage: float = 0.0
    building_percentage: float = 0.0
    vegetation_percentage: float = 0.0
    farmable_space_percentage: float = 0.0
    # When using AI masking (HF token): clear land, greenery, forest
    clear_land_percentage: Optional[float] = None
    greenery_percentage: Optional[float] = None
    forest_percentage: Optional[float] = None
    semantic_masking_used: bool = False


def get_token(authorization: Optional[str] = Header(None)) -> str:
    """Extract and validate Bearer token from Authorization header."""
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing token")
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(status_code=401, detail="Invalid authorization header")
    token = parts[1]
    payload = verify_token(token)
    if payload is None:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return token


@app.post("/login", response_model=LoginResponse)
def login(req: LoginRequest):
    """Login with username/password; returns JWT."""
    if not verify_user(req.username, req.password):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    token = create_access_token(data={"sub": req.username})
    return LoginResponse(access_token=token)


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(
    file: UploadFile = File(...),
    boundary: Optional[str] = Form(None),
    token: str = Depends(get_token),
):
    """Analyze uploaded image with real OpenCV processing; optional farm boundary (geo polygon) for clipping."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid image: file must be an image")

    try:
        contents = file.file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: could not read file - {str(e)}")

    if not contents:
        raise HTTPException(status_code=400, detail="Invalid image: empty file")

    try:
        cultivated_percentage, stress_percentage, image, vegetation_mask, farmable_mask, removed_pixels, original_resolution, processing_resolution = analyze_image(contents)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"OpenCV processing failed: {str(e)}")

    h, w = image.shape[:2]
    full_image_pixels = h * w
    total_pixels = full_image_pixels
    cultivated_pixels = int(cv2.countNonZero(vegetation_mask))
    use_boundary = False
    polygon_lat_lng = None
    pixels_inside_boundary = None
    pixels_outside_boundary = None
    percentage_inside_boundary = None
    percentage_outside_boundary = None

    if boundary and boundary.strip():
        try:
            polygon_lat_lng = json.loads(boundary)
            if isinstance(polygon_lat_lng, list) and len(polygon_lat_lng) >= 3:
                clipped_veg, cultivated_pixels, total_pixels = clip_cultivated_mask_by_boundary(
                    vegetation_mask, polygon_lat_lng
                )
                vegetation_mask = clipped_veg
                clipped_farmable, _, _ = clip_cultivated_mask_by_boundary(
                    farmable_mask, polygon_lat_lng
                )
                farmable_mask = clipped_farmable
                cultivated_percentage = (cultivated_pixels / total_pixels * 100.0) if total_pixels else 0.0
                cultivated_percentage = round(cultivated_percentage, 2)
                use_boundary = True
                pixels_inside_boundary = total_pixels
                pixels_outside_boundary = full_image_pixels - total_pixels
                percentage_inside_boundary = round(total_pixels / full_image_pixels * 100.0, 2) if full_image_pixels else 0.0
                percentage_outside_boundary = round(pixels_outside_boundary / full_image_pixels * 100.0, 2) if full_image_pixels else 0.0
            else:
                raise HTTPException(status_code=400, detail="Polygon must have at least 3 points")
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid boundary JSON")
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    non_cultivated_pixels = total_pixels - cultivated_pixels
    coverage_ratio = round(cultivated_pixels / total_pixels, 4) if total_pixels else 0.0

    contours, _ = cv2.findContours(vegetation_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_count = len(contours)

    conf = compute_deterministic_confidence(
        image, vegetation_mask, contour_count, total_pixels, cultivated_pixels
    )
    mask_confidence = conf["confidence"]
    clarity_score = conf["clarity_score"]
    fragmentation_score = conf["fragmentation_score"]
    edge_consistency_score = conf["edge_consistency_score"]
    area_plausibility_score = conf["area_plausibility_score"]
    exg_variance = conf["exg_variance"]

    vegetation_ratio = coverage_ratio
    laplacian_variance = float(cv2.Laplacian(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
    noise_ratio = round(removed_pixels / total_pixels, 4) if total_pixels else 0.0
    noise_score = max(0.0, min(1.0, 1.0 - noise_ratio))
    sharpness_score = round(min(laplacian_variance / 500.0, 1.0), 4)
    veg_balance_score = max(0.0, min(1.0, 1.0 - abs(0.5 - vegetation_ratio)))
    veg_balance_score = round(veg_balance_score, 4)

    is_blurry = laplacian_variance < 100
    trust_score_display = round(mask_confidence)
    advisory = get_advisory(cultivated_percentage, stress_percentage, trust_score_display)
    classification = _cultivation_classification(cultivated_percentage)
    analysis_history.append({
        "cultivated_percentage": cultivated_percentage,
        "stress_percentage": stress_percentage,
        "trust_score": mask_confidence,
        "advisory": advisory,
        "classification": classification,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })

    # --- Best masking: try AI semantic segmentation (free HF API) when token is set ---
    hf_token = os.environ.get("BUILDING_SEGMENTATION_HF_TOKEN") or os.environ.get("HF_TOKEN")
    semantic_masks = get_semantic_masks_ade20k(image, hf_token) if hf_token else None
    semantic_masking_used = semantic_masks is not None

    seg_bytes = get_segmentation_preview(image, vegetation_mask)
    seg_b64 = base64.b64encode(seg_bytes).decode("ascii")
    overlay_bgr = get_overlay_mask(image, vegetation_mask, farmable_mask)
    binary_b64 = base64.b64encode(get_binary_mask_png(farmable_mask)).decode("ascii")
    overlay_b64 = base64.b64encode(get_overlay_mask_png(overlay_bgr)).decode("ascii")
    land_use_bytes = get_landuse_mask_png(image, vegetation_mask, farmable_mask, semantic_masks)
    land_use_b64 = base64.b64encode(land_use_bytes).decode("ascii")

    # --- Land-use breakdown percentages (from AI semantic masks when available, else rule-based) ---
    if semantic_masks is not None and total_pixels:
        water_pixels = int(cv2.countNonZero(semantic_masks.get("water", np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8))))
        road_pixels = int(cv2.countNonZero(semantic_masks.get("road", np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8))))
        building_pixels = int(cv2.countNonZero(semantic_masks.get("building", np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8))))
        clear_land_pixels = int(cv2.countNonZero(semantic_masks.get("clear_land", np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8))))
        greenery_pixels = int(cv2.countNonZero(semantic_masks.get("greenery", np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8))))
        forest_pixels = int(cv2.countNonZero(semantic_masks.get("forest", np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8))))
        water_pct = round(water_pixels / total_pixels * 100, 2)
        road_pct = round(road_pixels / total_pixels * 100, 2)
        building_pct = round(building_pixels / total_pixels * 100, 2)
        clear_land_pct = round(clear_land_pixels / total_pixels * 100, 2)
        greenery_pct = round(greenery_pixels / total_pixels * 100, 2)
        forest_pct = round(forest_pixels / total_pixels * 100, 2)
        veg_pct = round(cultivated_pixels / total_pixels * 100, 2)
        farmable_space_pct = round(max(0, clear_land_pct + greenery_pct + forest_pct), 2)
    else:
        from image_processor import detect_water, detect_roads, detect_buildings
        water_mask = detect_water(image)
        road_mask = detect_roads(image)
        building_mask = detect_buildings(image, vegetation_mask)
        water_pixels = int(cv2.countNonZero(water_mask))
        road_pixels = int(cv2.countNonZero(road_mask))
        building_pixels = int(cv2.countNonZero(building_mask))
        water_pct = round(water_pixels / total_pixels * 100, 2) if total_pixels else 0.0
        road_pct = round(road_pixels / total_pixels * 100, 2) if total_pixels else 0.0
        building_pct = round(building_pixels / total_pixels * 100, 2) if total_pixels else 0.0
        clear_land_pct = None
        greenery_pct = None
        forest_pct = None
        veg_pct = round(cultivated_pixels / total_pixels * 100, 2) if total_pixels else 0.0
        farmable_space_pct = round(max(0, 100 - water_pct - road_pct - building_pct), 2) if total_pixels else 0.0

    # Polygon mask from cultivated area (findContours + approxPolyDP), JSON-serializable
    polygon_coordinates = get_polygons_from_cultivated_mask(vegetation_mask)
    polygon_overlay_bgr = overlay_polygons_on_image(image.copy(), polygon_coordinates)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    overlay_filename = f"cultivated_{ts}.png"
    overlay_path = OUTPUTS_DIR / overlay_filename
    cv2.imwrite(str(overlay_path), polygon_overlay_bgr)
    cultivated_mask_image_path = f"/outputs/{overlay_filename}"

    return AnalyzeResponse(
        cultivated_percentage=cultivated_percentage,
        stress_percentage=stress_percentage,
        trust_score=mask_confidence,
        mask_confidence=mask_confidence,
        advisory=advisory,
        is_blurry=is_blurry,
        classification=classification,
        total_pixels=total_pixels,
        cultivated_pixels=cultivated_pixels,
        non_cultivated_pixels=non_cultivated_pixels,
        coverage_ratio=coverage_ratio,
        contour_count=contour_count,
        sharpness_score=sharpness_score,
        veg_balance_score=veg_balance_score,
        fragmentation_score=fragmentation_score,
        noise_score=noise_score,
        laplacian_variance=laplacian_variance,
        vegetation_ratio=vegetation_ratio,
        noise_ratio=noise_ratio,
        clarity_score=clarity_score,
        edge_consistency_score=edge_consistency_score,
        area_plausibility_score=area_plausibility_score,
        exg_variance=exg_variance,
        original_resolution=original_resolution,
        processing_resolution=processing_resolution,
        segmentation_image_base64=seg_b64,
        binary_mask_base64=binary_b64,
        overlay_mask_base64=overlay_b64,
        polygon_coordinates=polygon_coordinates,
        cultivated_mask_image_path=cultivated_mask_image_path,
        within_boundary=use_boundary,
        full_image_pixels=full_image_pixels,
        pixels_inside_boundary=pixels_inside_boundary,
        pixels_outside_boundary=pixels_outside_boundary,
        percentage_inside_boundary=percentage_inside_boundary,
        percentage_outside_boundary=percentage_outside_boundary,
        land_use_mask_base64=land_use_b64,
        water_percentage=water_pct,
        road_percentage=road_pct,
        building_percentage=building_pct,
        vegetation_percentage=veg_pct,
        farmable_space_percentage=farmable_space_pct,
        clear_land_percentage=clear_land_pct if semantic_masking_used else None,
        greenery_percentage=greenery_pct if semantic_masking_used else None,
        forest_percentage=forest_pct if semantic_masking_used else None,
        semantic_masking_used=semantic_masking_used,
    )


@app.get("/history")
def history(token: str = Depends(get_token)) -> List[dict]:
    """Return last 5 analyses. Protected by JWT."""
    return analysis_history[-5:][::-1]


@app.get("/weather")
def get_weather(city: str, token: str = Depends(get_token)):
    """Get weather data for a city using OpenWeatherMap API."""
    import requests
    WEATHER_API_KEY = "42a28b0639a88def3bb8150c4d61f466"
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_API_KEY}&units=metric"
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            error_data = response.json() if response.text else {}
            raise HTTPException(status_code=400, detail=f"Weather API error: {error_data.get('message', 'Unknown error')}")
        data = response.json()
        return {
            "success": True,
            "name": data.get("name"),
            "temp": data.get("main", {}).get("temp"),
            "feels_like": data.get("main", {}).get("feels_like"),
            "humidity": data.get("main", {}).get("humidity"),
            "pressure": data.get("main", {}).get("pressure"),
            "wind_speed": data.get("wind", {}).get("speed"),
            "weather_main": data.get("weather", [{}])[0].get("main") if data.get("weather") else "Unknown",
            "weather_description": data.get("weather", [{}])[0].get("description") if data.get("weather") else "unknown",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/forecast")
def get_forecast(city: str, token: str = Depends(get_token)):
    """Get 5-day forecast for a city."""
    import requests
    WEATHER_API_KEY = "42a28b0639a88def3bb8150c4d61f466"
    try:
        url = f"https://api.openweathermap.org/data/2.5/forecast?q={city}&appid={WEATHER_API_KEY}&units=metric"
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            return {"success": False, "detail": "Forecast unavailable"}
        data = response.json()
        # Process data to get daily forecasts (one per day)
        daily = []
        seen_dates = set()
        for item in data.get("list", []):
            date = item.get("dt_txt").split(" ")[0]
            if date not in seen_dates:
                seen_dates.add(date)
                daily.append({
                    "date": date,
                    "temp": item.get("main", {}).get("temp"),
                    "weather_main": item.get("weather", [{}])[0].get("main"),
                    "weather_description": item.get("weather", [{}])[0].get("description"),
                    "humidity": item.get("main", {}).get("humidity"),
                })
                if len(daily) >= 5: break
        return {"success": True, "city": data.get("city", {}).get("name"), "forecast": daily}
    except Exception as e:
        return {"success": False, "detail": str(e)}


@app.get("/forecast-by-coords")
def get_forecast_by_coords(lat: float, lon: float, token: str = Depends(get_token)):
    """Get 5-day forecast by coordinates."""
    import requests
    WEATHER_API_KEY = "42a28b0639a88def3bb8150c4d61f466"
    try:
        url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={WEATHER_API_KEY}&units=metric"
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            return {"success": False, "detail": "Forecast unavailable"}
        data = response.json()
        daily = []
        seen_dates = set()
        for item in data.get("list", []):
            date = item.get("dt_txt").split(" ")[0]
            if date not in seen_dates:
                seen_dates.add(date)
                daily.append({
                    "date": date,
                    "temp": item.get("main", {}).get("temp"),
                    "weather_main": item.get("weather", [{}])[0].get("main"),
                    "weather_description": item.get("weather", [{}])[0].get("description"),
                    "humidity": item.get("main", {}).get("humidity"),
                })
                if len(daily) >= 5: break
        return {"success": True, "city": data.get("city", {}).get("name"), "forecast": daily}
    except Exception as e:
        return {"success": False, "detail": str(e)}


@app.get("/weather-by-coords")
def get_weather_by_coords(
    lat: float = Query(..., description="Latitude"),
    lon: float = Query(..., description="Longitude"),
    token: str = Depends(get_token),
):
    """Get weather data by coordinates using OpenWeatherMap API. Auto-detects city name."""
    WEATHER_API_KEY = "42a28b0639a88def3bb8150c4d61f466"
    if not WEATHER_API_KEY:
        raise HTTPException(status_code=500, detail="Weather API key not configured")
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={WEATHER_API_KEY}&units=metric"
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            error_data = response.json() if response.text else {}
            raise HTTPException(
                status_code=400,
                detail=f"Weather API error: {error_data.get('message', f'HTTP {response.status_code}')}"
            )
        data = response.json()
        if data.get("cod") == 200:
            return {
                "success": True,
                "name": data.get("name", "Unknown"),
                "country": data.get("sys", {}).get("country", ""),
                "lat": lat,
                "lon": lon,
                "temp": data.get("main", {}).get("temp"),
                "feels_like": data.get("main", {}).get("feels_like"),
                "humidity": data.get("main", {}).get("humidity"),
                "pressure": data.get("main", {}).get("pressure"),
                "wind_speed": data.get("wind", {}).get("speed"),
                "weather_main": data.get("weather", [{}])[0].get("main") if data.get("weather") else "Unknown",
                "weather_description": data.get("weather", [{}])[0].get("description") if data.get("weather") else "unknown",
            }
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Weather API error: {data.get('message', 'Unknown error')}"
            )
    except HTTPException:
        raise
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=500, detail="Weather API request timed out")
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch weather: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


def _compute_plot_suitability(
    farmable_space_pct: float,
    water_pct: float,
    building_pct: float,
    cultivated_pct: float,
) -> tuple:
    """
    Returns (suitability, message, suggestions_list).
    suitability: "suitable" | "marginal" | "not_suitable"
    """
    suggestions = []
    # Not suitable: mostly built-up or water, very little farmable land
    if farmable_space_pct < 15:
        return (
            "not_suitable",
            "This plot has very little farmable space. Not recommended for agriculture.",
            ["Consider using the area for non-farming use or soil improvement first."],
        )
    if building_pct > 50:
        return (
            "not_suitable",
            "Most of the area is built-up. Not suitable for crop cultivation.",
            ["If you have small patches, consider kitchen garden or container farming."],
        )
    # Marginal: limited space or water
    if farmable_space_pct < 35 or water_pct < 2:
        msg = "Marginal conditions: limited farmable space or water."
        if water_pct < 2:
            suggestions.append("Consider drought-resistant crops (millet, sorghum, chickpea).")
        if farmable_space_pct < 35:
            suggestions.append("Focus on high-value or short-duration crops to use space efficiently.")
        suggestions.append("Improving irrigation or soil moisture can expand options.")
        return ("marginal", msg, suggestions)
    # Suitable
    msg = "This plot is suitable for agriculture."
    if water_pct >= 8:
        suggestions.append("Good water availability — suitable for paddy, sugarcane, or vegetables needing regular irrigation.")
    elif water_pct >= 3:
        suggestions.append("Moderate water — pulses, maize, and most vegetables can do well.")
    else:
        suggestions.append("Limited water — prefer drought-tolerant crops and efficient irrigation.")
    if cultivated_pct < 30:
        suggestions.append("Significant uncultivated area — consider cover crops or expanding cultivation.")
    if farmable_space_pct >= 60:
        suggestions.append("Large farmable area — you can plan multiple crops or rotation.")
    return ("suitable", msg, suggestions)


@app.get("/plot-suggestion")
def get_plot_suggestion(
    lat: float = Query(..., description="Latitude of the plot"),
    lon: float = Query(..., description="Longitude of the plot"),
    cultivated_pct: float = Query(50.0, description="Currently cultivated percentage (from image analysis or estimate)"),
    water_pct: float = Query(5.0, description="Water detected percentage"),
    farmable_space_pct: float = Query(50.0, description="Farmable space percentage"),
    building_pct: float = Query(0.0, description="Building/urban percentage"),
    soil_moisture: Optional[str] = Query(None, description="Soil moisture: low / medium / high"),
    soil_type: Optional[str] = Query(None, description="Soil type: clay, loam, sandy, etc."),
    soil_ph: Optional[str] = Query(None, description="Soil pH: acidic / neutral / alkaline"),
    rainfall_expected: Optional[str] = Query(None, description="Expected rainfall: low / medium / high"),
    token: str = Depends(get_token),
):
    """
    AI-style suggestion for a selected plot: suitability for agriculture, short message,
    bullet suggestions, current weather, and top crop recommendations.
    Use after image analysis (pass cultivated_pct, water_pct, farmable_space_pct, building_pct)
    or with manual estimates. Lat/lon required for weather and season.
    """
    now = datetime.now()
    month = now.month

    # 1) Suitability from land-use
    suitability, suitability_message, suggestions = _compute_plot_suitability(
        farmable_space_pct, water_pct, building_pct, cultivated_pct
    )

    # 2) Weather at plot location
    weather_data = None
    WEATHER_API_KEY = os.environ.get("WEATHER_API_KEY") or "42a28b0639a88def3bb8150c4d61f466"
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={WEATHER_API_KEY}&units=metric"
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            data = r.json()
            if data.get("cod") == 200:
                weather_data = {
                    "name": data.get("name", "Unknown"),
                    "country": data.get("sys", {}).get("country", ""),
                    "temp": data.get("main", {}).get("temp"),
                    "feels_like": data.get("main", {}).get("feels_like"),
                    "humidity": data.get("main", {}).get("humidity"),
                    "weather_main": data.get("weather", [{}])[0].get("main") if data.get("weather") else "Unknown",
                    "weather_description": data.get("weather", [{}])[0].get("description") if data.get("weather") else "unknown",
                }
    except Exception:
        pass

    temp = weather_data["temp"] if weather_data and weather_data.get("temp") is not None else 25.0
    humidity = weather_data["humidity"] if weather_data and weather_data.get("humidity") is not None else 60.0

    # 3) Crop recommendations
    crop_result = recommend_crops(
        temperature=float(temp),
        humidity=float(humidity),
        latitude=lat,
        month=month,
        cultivated_pct=cultivated_pct,
        water_pct=water_pct,
        farmable_space_pct=farmable_space_pct,
        soil_moisture=soil_moisture,
        soil_type=soil_type,
        soil_ph=soil_ph,
        rainfall_expected=rainfall_expected,
    )

    primary = crop_result.get("primary_crop")
    top_crops = (crop_result.get("highly_recommended") or [])[:5]

    return {
        "success": True,
        "suitability": suitability,
        "suitability_message": suitability_message,
        "suggestions": suggestions,
        "weather": weather_data,
        "primary_crop": primary,
        "top_crops": top_crops,
        "season": crop_result.get("season_display"),
        "summary": crop_result.get("summary"),
        "indicators_used": crop_result.get("indicators_used", []),
    }


@app.get("/recommend-crops")
def get_crop_recommendations(
    temp: float = Query(..., description="Temperature in Celsius"),
    humidity: float = Query(..., description="Humidity percentage"),
    lat: float = Query(..., description="Latitude"),
    cultivated_pct: float = Query(50.0, description="Currently cultivated percentage"),
    water_pct: float = Query(5.0, description="Water detected percentage"),
    farmable_space_pct: float = Query(50.0, description="Farmable space percentage"),
    soil_moisture: Optional[str] = Query(None, description="Soil moisture: low / medium / high, or 0-100"),
    soil_type: Optional[str] = Query(None, description="Soil type: clay, loam, sandy, black alluvial, red laterite, etc."),
    soil_ph: Optional[str] = Query(None, description="Soil pH: acidic / neutral / alkaline"),
    rainfall_expected: Optional[str] = Query(None, description="Expected rainfall: low / medium / high"),
    token: str = Depends(get_token),
):
    """Get crop recommendations from real agronomic data (temp, humidity, season, water, space, soil, rainfall)."""
    now = datetime.now()
    month = now.month
    result = recommend_crops(
        temperature=temp,
        humidity=humidity,
        latitude=lat,
        month=month,
        cultivated_pct=cultivated_pct,
        water_pct=water_pct,
        farmable_space_pct=farmable_space_pct,
        soil_moisture=soil_moisture,
        soil_type=soil_type,
        soil_ph=soil_ph,
        rainfall_expected=rainfall_expected,
    )
    return {"success": True, **result}


@app.get("/health")
def health():
    return {
        "status": "active",
        "model_version": "2.0",
        "security_layer": "JWT enabled",
        "features": [
            "image_analysis",
            "weather",
            "crop_recommendations",
            "land_use_detection",
            "plot_suggestion",
            "demo_image",
        ],
    }
