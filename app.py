#!/usr/bin/env python3
"""
FastAPI Backend: Location-Aware HAZUS Economic Loss (EL), Smart Evacuation & Dynamic ST-GNN Risk Mapping.
"""

import math
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional
import numpy as np
import torch
import torch.nn as nn
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

_v_erf = np.vectorize(math.erf)

def standard_normal_cdf(x):
    return 0.5 * (1.0 + _v_erf(x / np.sqrt(2.0)))

# ==========================================
# 1. Asset & Hub Configuration
# ==========================================
MY_URL = "seismicinteractive.netlify.app" #"127.0.0.1"

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR if (SCRIPT_DIR / "web_grid_features.npz").exists() else SCRIPT_DIR.parent

FEATURES_NPZ = DATA_DIR / "web_grid_features.npz"
MODEL_PATH = DATA_DIR / "st_gnn_model.pt"

# Expanded Regional Network (LA Core + Perimeter Safe Havens)
EMERGENCY_HUBS = [
    # LA Urban Core
    {"name": "LA General Medical Center (LAC+USC)", "lat": 34.0578, "lon": -118.2068, "type": "Hospital"},
    {"name": "Ronald Reagan UCLA Medical Center", "lat": 34.0664, "lon": -118.4455, "type": "Hospital"},
    {"name": "Cedars-Sinai Medical Center", "lat": 34.0751, "lon": -118.3802, "type": "Hospital"},
    {"name": "Harbor-UCLA Medical Center", "lat": 33.8294, "lon": -118.2982, "type": "Hospital"},
    {"name": "Long Beach Memorial Care", "lat": 33.8078, "lon": -118.1883, "type": "Hospital"},
    {"name": "Huntington Hospital (Pasadena)", "lat": 34.1352, "lon": -118.1542, "type": "Hospital"},
    {"name": "Providence Saint Joseph Medical Center", "lat": 34.1565, "lon": -118.3228, "type": "Hospital"},
    {"name": "LA Convention Center Shelter", "lat": 34.0403, "lon": -118.2696, "type": "Evacuation Shelter"},
    {"name": "Rose Bowl Staging Center", "lat": 34.1613, "lon": -118.1676, "type": "Evacuation Shelter"},
    {"name": "Dignity Health Sports Park Hub", "lat": 33.8644, "lon": -118.2611, "type": "Evacuation Shelter"},
    {"name": "SoFi Stadium Relief Hub", "lat": 33.9535, "lon": -118.3390, "type": "Evacuation Shelter"},
    
    # Outer Perimeter / Safe Havens (For M6.3+ Quakes in LA)
    {"name": "Lancaster National Guard Armory", "lat": 34.6981, "lon": -118.1366, "type": "Evacuation Shelter"},
    {"name": "Santa Clarita Valley Sports Complex", "lat": 34.4262, "lon": -118.5283, "type": "Evacuation Shelter"},
    {"name": "Anaheim Convention Center Shelter", "lat": 33.8003, "lon": -117.9200, "type": "Evacuation Shelter"},
    {"name": "UC Irvine Medical Center", "lat": 33.7883, "lon": -117.8911, "type": "Hospital"},
    {"name": "Loma Linda University Medical Center", "lat": 34.0489, "lon": -117.2641, "type": "Hospital"},
    {"name": "National Orange Show Event Center (San Bernardino)", "lat": 34.0928, "lon": -117.2917, "type": "Evacuation Shelter"},
    {"name": "Palm Springs Convention Relief Center", "lat": 33.8236, "lon": -116.5393, "type": "Evacuation Shelter"},
    {"name": "Ventura County Fairgrounds Relief Hub", "lat": 34.2748, "lon": -119.3001, "type": "Evacuation Shelter"}
]

def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2.0) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2.0) ** 2
    return r * 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))

# ==========================================
# 2. PyTorch ST-GNN Architecture
# ==========================================

class GraphConvLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        deg = adj.sum(dim=-1, keepdim=True).clamp(min=1.0)
        out = self.fc(torch.matmul(adj / deg, x))
        return torch.relu(out)

class EarthquakeSTGNN(nn.Module):
    def __init__(self, in_channels: int = 4, hidden_dim: int = 32):
        super().__init__()
        self.gcn1 = GraphConvLayer(in_channels, hidden_dim)
        self.temp_conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(3, 1), padding=(1, 0))
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_dim, 16), nn.ReLU(),
            nn.Linear(16, 1), nn.Sigmoid()
        )

    def forward(self, x, adj):
        B, T, N, C = x.shape
        spatial_out = self.gcn1(x.view(B * T, N, C), adj).view(B, T, N, -1).permute(0, 3, 1, 2)
        temp_out = torch.relu(self.temp_conv(spatial_out))
        return self.fc_out(temp_out[:, :, -1, :].permute(0, 2, 1)).squeeze(-1)

state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    data = np.load(FEATURES_NPZ)
    model = EarthquakeSTGNN(in_channels=4, hidden_dim=32)
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    except TypeError:
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        
    model.eval()

    state.update({
        "model": model,
        "adj_tensor": torch.tensor(data["adj_matrix"].astype(np.float32)),
        "coords": data["coords"].astype(np.float32),
        "fault_decay": data["fault_decay"].astype(np.float32),
        "norm_dens": data["norm_dens"].astype(np.float32),
        "vs30_amp": data["vs30_amp"].astype(np.float32)
    })
    yield
    state.clear()

app = FastAPI(title="Earthquake Hazard & Safety API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=[MY_URL], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# ==========================================
# 3. Schemas & Endpoints
# ==========================================

class ScenarioRequest(BaseModel):
    epicenter_x: float
    epicenter_y: float
    magnitude: float = Field(..., ge=4.0, le=9.5)
    user_x: Optional[float] = None
    user_y: Optional[float] = None
    time_steps: int = 10

class PersonalImpact(BaseModel):
    in_hazard_zone: bool
    distance_to_epicenter_km: float
    critical_impact_probability: str
    structural_collapse_risk: str
    predicted_health_outcomes: List[str]
    escape_feasibility: str

class SafeFacility(BaseModel):
    name: str
    type: str
    lat: float
    lon: float
    distance_from_user_km: float
    is_outside_impact_zone: bool
    safety_rating: str

class SimulationResponse(BaseModel):
    radius_of_effect_km: float
    damaged_buildings: int
    economic_loss: str
    personal_impact: Optional[PersonalImpact]
    safe_evacuation_centers: List[SafeFacility]
    predictions: List[dict]

@app.post("/predict", response_model=SimulationResponse)
def predict_risk(req: ScenarioRequest):
    coords = state["coords"]
    num_nodes = len(coords)

    # 1. Attenuation & Dynamic Shaking Intensity (PGA)
    dists = haversine_km(coords[:, 1], coords[:, 0], req.epicenter_y, req.epicenter_x)
    
    # Fault rupture length approximation (Joyner-Boore distance equivalent)
    r_jb = np.maximum(dists - 0.15 * (10 ** (0.4 * req.magnitude - 1.2)), 1.0)
    r_hypo = np.sqrt(r_jb**2 + 8.0**2)
    
    # Calibrated GMPE for SoCal regional attenuation and basin effects
    ln_pga = -0.15 + 0.92 * (req.magnitude - 6.0) - 0.82 * np.log(r_hypo) + 0.25 * state["vs30_amp"]
    pga_g = np.exp(ln_pga)

    # 2. Dynamic GNN Tensor Input
    wave_att = np.clip(pga_g / 1.5, 0.0, 1.0)
    scenario_tensor = np.zeros((1, req.time_steps, num_nodes, 4), dtype=np.float32)
    for t in range(req.time_steps):
        scenario_tensor[0, t, :, 0] = state["fault_decay"]
        scenario_tensor[0, t, :, 1] = state["norm_dens"]
        scenario_tensor[0, t, :, 2] = state["vs30_amp"]
        scenario_tensor[0, t, :, 3] = wave_att * ((t + 1) / req.time_steps)

    with torch.no_grad():
        raw_preds = state["model"](torch.tensor(scenario_tensor), state["adj_tensor"]).squeeze(0).numpy()

    # Map Ground Acceleration directly to Normalized Hazard Index [0.0 - 1.0]
    pga_hazard_index = np.clip(pga_g / 0.40, 0.0, 1.0)
    calibrated_risk = np.clip((pga_hazard_index * 0.75) + (raw_preds * 0.25), 0.0, 0.999)

    # Effective Damage Radius
    radius_km = float(10 ** (0.45 * req.magnitude - 1.1))

    # HAZUS Structural Loss & Fragility Curves
    damage_ratio = standard_normal_cdf((np.log(np.clip(pga_g, 0.001, 5.0)) - np.log(0.18)) / 0.50)
    node_asset_value = state["norm_dens"] * 2.5e10 + 5.0e8
    total_el_usd = float(np.sum(node_asset_value * damage_ratio))

    node_building_count = state["norm_dens"] * 35000.0 + 1000.0
    damaged_buildings = int(np.sum(node_building_count * damage_ratio))

    if total_el_usd >= 1e12:
        el_str = f"${total_el_usd / 1e12:.2f} Trillion"
    elif total_el_usd >= 1e9:
        el_str = f"${total_el_usd / 1e9:.2f} Billion"
    elif total_el_usd >= 1e6:
        el_str = f"${total_el_usd / 1e6:.2f} Million"
    else:
        el_str = f"${total_el_usd:,.0f}"

    # 3. Personal Hazard Profile (Ground Motion Hazard Index)
    personal_impact = None
    pga_user = 0.0
    if req.user_x is not None and req.user_y is not None:
        user_dist = haversine_km(req.user_y, req.user_x, req.epicenter_y, req.epicenter_x)
        in_zone = bool(user_dist <= radius_km)

        # Local PGA calculation at user location
        user_r_jb = max(user_dist - 0.15 * (10 ** (0.4 * req.magnitude - 1.2)), 1.0)
        user_r_hypo = np.sqrt(user_r_jb**2 + 8.0**2)
        pga_user = float(np.exp(-0.15 + 0.92 * (req.magnitude - 6.0) - 0.82 * np.log(user_r_hypo)))

        # Ground Motion Hazard Score
        crit_prob = float(np.clip(pga_user / 0.35, 0.0, 1.0) * 100.0)

        if pga_user >= 0.30:
            collapse = "Severe Risk: Violent Shaking (MMI VIII+), High Risk of Structural Failure in Non-Retrofitted Buildings"
            health_outcomes = [
                "Severe Life-Threatening Hazards: Structural Collapse & Heavy Debris Ingress",
                "High Risk from Unreinforced Masonry Failures",
                "Secondary Hazards: Infrastructure Ruptures, Fires, & Hazardous Dust Inhalation"
            ]
            escape = "Extremely Impaired: Move to Local Staging / Open Areas Immediately"
        elif pga_user >= 0.15:
            collapse = "Moderate Hazard: Very Strong Shaking (MMI VII), Plaster/Tile Cracking & Content Dislodgement"
            health_outcomes = [
                "Moderate Risk: Injuries from Unanchored Furniture & Shattered Glass",
                "Secondary Risk: Gas Line Breaches & Falling Fixtures"
            ]
            escape = "Impaired: Exercise High Caution During Egress"
        elif pga_user >= 0.05:
            collapse = "Light Hazard: Noticeable Shaking (MMI V-VI), Small Items Dislodged"
            health_outcomes = ["Low Risk: Minor contusions or scrapes from loose objects"]
            escape = "Safe & Clear: Normal Egress Paths Functional"
        else:
            collapse = "Minimal Shaking: Feeble Vibration"
            health_outcomes = ["No Health Hazards Anticipated"]
            escape = "Completely Unobstructed"

        personal_impact = PersonalImpact(
            in_hazard_zone=in_zone,
            distance_to_epicenter_km=round(float(user_dist), 2),
            critical_impact_probability=f"{crit_prob:.1f}%",
            structural_collapse_risk=collapse,
            predicted_health_outcomes=health_outcomes,
            escape_feasibility=escape
        )

    # 4. Smart Evacuation Routing Strategy
    user_lat = req.user_y if req.user_y is not None else req.epicenter_y
    user_lon = req.user_x if req.user_x is not None else req.epicenter_x

    evaluated_hubs = []
    for hub in EMERGENCY_HUBS:
        dist_from_epicenter = haversine_km(hub["lat"], hub["lon"], req.epicenter_y, req.epicenter_x)
        dist_from_user = haversine_km(hub["lat"], hub["lon"], user_lat, user_lon)
        is_safe_zone = bool(dist_from_epicenter > radius_km)

        evaluated_hubs.append(SafeFacility(
            name=hub["name"],
            type=hub["type"],
            lat=hub["lat"],
            lon=hub["lon"],
            distance_from_user_km=round(float(dist_from_user), 2),
            is_outside_impact_zone=is_safe_zone,
            safety_rating="Optimal (Outside Shake Zone)" if is_safe_zone else "Functional Local Hub"
        ))

    # Dynamic sorting based on local shaking severity
    def compute_hub_score(hub_facility: SafeFacility):
        if pga_user < 0.25:
            return hub_facility.distance_from_user_km
        else:
            safety_penalty = 0.0 if hub_facility.is_outside_impact_zone else 500.0
            return hub_facility.distance_from_user_km + safety_penalty

    evaluated_hubs.sort(key=compute_hub_score)
    results = [{"coords": coords[i].tolist(), "predicted_risk": round(float(calibrated_risk[i]), 4)} for i in range(num_nodes)]

    return SimulationResponse(
        radius_of_effect_km=round(radius_km, 1),
        damaged_buildings=damaged_buildings,
        economic_loss=el_str,
        personal_impact=personal_impact,
        safe_evacuation_centers=evaluated_hubs[:6],
        predictions=results
    )

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=10000, reload=True)
