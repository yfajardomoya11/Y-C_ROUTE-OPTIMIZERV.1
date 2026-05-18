import uvicorn
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import HTMLResponse
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer
from pydantic import BaseModel
from jose import jwt, JWTError
from ortools.constraint_solver import routing_enums_pb2, pywrapcp
from typing import List
import math
import asyncio
import httpx

# --- CONFIGURACIÓN ---
SECRET_KEY = "YC_ROUTE_OPTIMIZER_SECRET_2026"
ALGORITHM = "HS256"

VALID_USERS = {
    "admin": "admin123",
    "DaniPruebas-16": "daniela16"
}

app = FastAPI()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
PALMARES_DEPOT = [10.0605, -84.4372]


class Delivery(BaseModel):
    id: int
    lat: float
    lon: float
    descripcion: str


class RouteRequest(BaseModel):
    deliveries: List[Delivery]
    num_vehicles: int = 5
    capacity_per_vehicle: int = 18


# --- AUTH ---
def verify_token(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Token inválido")
        return username
    except JWTError:
        raise HTTPException(status_code=401, detail="Token inválido o expirado")


@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    password = VALID_USERS.get(form_data.username)
    if password is None or form_data.password != password:
        raise HTTPException(status_code=401, detail="Credenciales inválidas")
    token = jwt.encode({"sub": form_data.username}, SECRET_KEY, algorithm=ALGORITHM)
    return {"access_token": token, "token_type": "bearer"}


# --- VRP ---
def haversine_meters(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return int(2 * R * math.asin(math.sqrt(a)))


def solve_vrp(deliveries, num_vehicles=5, capacity_per_vehicle=25):
    print(f"--> Optimizando {len(deliveries)} paradas con {num_vehicles} vehículos...")
    locations = [PALMARES_DEPOT] + [[d.lat, d.lon] for d in deliveries]
    n = len(locations)

    manager = pywrapcp.RoutingIndexManager(n, num_vehicles, 0)
    routing = pywrapcp.RoutingModel(manager)

    dist_matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                dist_matrix[i][j] = haversine_meters(
                    locations[i][0], locations[i][1],
                    locations[j][0], locations[j][1]
                )

    def dist_cb(f_idx, t_idx):
        return dist_matrix[manager.IndexToNode(f_idx)][manager.IndexToNode(t_idx)]

    transit_cb = routing.RegisterTransitCallback(dist_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_cb)

    def demand_cb(idx):
        node = manager.IndexToNode(idx)
        return 0 if node == 0 else 1

    demand_cb_idx = routing.RegisterUnaryTransitCallback(demand_cb)
    routing.AddDimensionWithVehicleCapacity(
        demand_cb_idx, 0,
        [capacity_per_vehicle] * num_vehicles,
        True, "Capacity"
    )

    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_params.time_limit.seconds = 10

    sol = routing.SolveWithParameters(search_params)

    res = []
    if sol:
        for v in range(num_vehicles):
            idx = routing.Start(v)
            route_nodes = []
            route_coords = []
            total_dist = 0

            while not routing.IsEnd(idx):
                node = manager.IndexToNode(idx)
                route_nodes.append(node)
                route_coords.append(locations[node])
                next_idx = sol.Value(routing.NextVar(idx))
                if not routing.IsEnd(next_idx):
                    total_dist += dist_matrix[node][manager.IndexToNode(next_idx)]
                idx = next_idx

            route_coords.append(locations[0])
            stops = [c for i, c in enumerate(route_coords[:-1]) if i > 0]
            if len(stops) == 0:
                continue

            # ---- Google Maps URL (formato dirección múltiple) ----
            # origin y destination = depósito
            origin = f"{PALMARES_DEPOT[0]},{PALMARES_DEPOT[1]}"

            waypoints = route_coords[1:-1]   # paradas intermedias (sin depot)
            wps = waypoints[:23]             # Google Maps acepta hasta 23 waypoints

            # Encode "/" en coordenadas para evitar confusión en la URL
            # Se usa el separador "|" entre waypoints
            wps_str = "|".join(f"{c[0]},{c[1]}" for c in wps)

            # Construir URL con urllib para asegurar encoding correcto
            from urllib.parse import quote
            maps_url = (
                "https://www.google.com/maps/dir/?api=1"
                f"&origin={origin}"
                f"&destination={origin}"
                f"&waypoints={quote(wps_str, safe=',|')}"
                "&travelmode=driving"
            )

            km_est = round(total_dist / 1000, 1)

            stop_info = []
            for node in route_nodes[1:]:
                d_idx = node - 1
                if 0 <= d_idx < len(deliveries):
                    stop_info.append({
                        "lat": deliveries[d_idx].lat,
                        "lon": deliveries[d_idx].lon,
                        "nombre": deliveries[d_idx].descripcion
                    })

            res.append({
                "vehicle": v + 1,
                "route": route_coords,
                "km": km_est,
                "maps_url": maps_url,
                "stops": len(stops),
                "stop_info": stop_info
            })

    return res


# --- OSRM ---
OSRM_BASE = "https://router.project-osrm.org/route/v1/driving"

async def get_osrm_geometry(coords: list) -> list:
    if len(coords) < 2:
        return coords

    coords_str = ";".join(f"{c[1]},{c[0]}" for c in coords)
    url = f"{OSRM_BASE}/{coords_str}?overview=full&geometries=geojson&steps=false"

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.get(url)
            r.raise_for_status()
            data = r.json()

        if data.get("code") != "Ok" or not data.get("routes"):
            print(f"  [OSRM] Sin ruta válida, usando línea recta.")
            return coords

        geom = data["routes"][0]["geometry"]["coordinates"]
        road_coords = [[pt[1], pt[0]] for pt in geom]

        distance_m = data["routes"][0]["distance"]
        km_real = round(distance_m / 1000, 1)

        return road_coords, km_real

    except Exception as e:
        print(f"  [OSRM] Error: {e} — usando línea recta como fallback.")
        return coords, None


@app.post("/optimize")
async def optimize(req: RouteRequest, username: str = Depends(verify_token)):
    if not req.deliveries:
        raise HTTPException(status_code=400, detail="No hay paradas para optimizar.")
    try:
        loop = asyncio.get_event_loop()
        routes = await loop.run_in_executor(
            None, solve_vrp, req.deliveries, req.num_vehicles, req.capacity_per_vehicle
        )

        async def enrich_route(rt):
            raw_coords = rt["route"]
            result = await get_osrm_geometry(raw_coords)

            if isinstance(result, tuple):
                road_coords, km_real = result
                rt["route"] = road_coords
                if km_real is not None:
                    rt["km"] = km_real
                    rt["km_source"] = "osrm"
                else:
                    rt["km_source"] = "haversine"
            else:
                rt["route"] = result
                rt["km_source"] = "haversine"

            return rt

        enriched = await asyncio.gather(*[enrich_route(rt) for rt in routes])

        return {"routes": enriched, "total_stops": len(req.deliveries)}

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/", response_class=HTMLResponse)
async def gui():
    return r"""
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Y&C Route Optimizer</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Orbitron:wght@400;700;900&family=Rajdhani:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg:        #040b14;
            --bg2:       #071525;
            --bg3:       #0a1e35;
            --border:    #0d2d4a;
            --border2:   #1a4a70;
            --cyan:      #00d4ff;
            --cyan2:     #00a8d4;
            --green:     #00ff9d;
            --yellow:    #ffe033;
            --red:       #ff3366;
            --purple:    #bd5fff;
            --orange:    #ff8c42;
            --text:      #c8dce8;
            --text2:     #5a7a90;
            --text3:     #2a4a60;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }

        body {
            background: var(--bg);
            color: var(--text);
            font-family: 'Rajdhani', sans-serif;
            font-size: 15px;
            min-height: 100vh;
            overflow-x: hidden;
        }

        #dash::before {
            content: '';
            position: fixed;
            inset: 0;
            background-image:
                linear-gradient(rgba(0,212,255,0.025) 1px, transparent 1px),
                linear-gradient(90deg, rgba(0,212,255,0.025) 1px, transparent 1px);
            background-size: 40px 40px;
            pointer-events: none;
            z-index: 0;
        }

        /* === LOGIN === */
        #login {
            position: fixed;
            inset: 0;
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 1000;
            overflow: hidden;
            background: #020c18;
        }

        #login-canvas {
            position: absolute;
            inset: 0;
            width: 100%;
            height: 100%;
        }

        #login::after {
            content: '';
            position: absolute;
            inset: 0;
            background: radial-gradient(ellipse 70% 70% at 50% 50%,
                transparent 0%, rgba(2,12,24,0.55) 60%, rgba(2,12,24,0.92) 100%);
            pointer-events: none;
        }

        .login-panel {
            position: relative;
            z-index: 10;
            display: flex;
            width: 900px;
            max-width: 96vw;
            min-height: 520px;
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid rgba(0,212,255,0.18);
            box-shadow: 0 0 80px rgba(0,212,255,0.07), 0 40px 80px rgba(0,0,0,0.6);
        }

        .login-left {
            flex: 1;
            background: linear-gradient(160deg, rgba(0,30,55,0.92) 0%, rgba(0,15,30,0.97) 100%);
            border-right: 1px solid rgba(0,212,255,0.12);
            padding: 3rem 2.5rem;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }

        .ll-logo {
            display: flex;
            align-items: center;
            gap: 0.8rem;
        }

        .ll-icon {
            width: 44px;
            height: 44px;
            border-radius: 10px;
            background: rgba(0,212,255,0.12);
            border: 1px solid rgba(0,212,255,0.3);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.3rem;
            flex-shrink: 0;
        }

        .ll-brand {
            font-family: 'Orbitron', monospace;
            font-size: 0.9rem;
            font-weight: 900;
            color: var(--cyan);
            letter-spacing: 0.1em;
            line-height: 1.2;
        }

        .ll-brand-sub {
            font-family: 'Space Mono', monospace;
            font-size: 0.45rem;
            color: var(--text3);
            letter-spacing: 0.25em;
            margin-top: 0.2rem;
        }

        .ll-hero {
            flex: 1;
            display: flex;
            flex-direction: column;
            justify-content: center;
            padding: 2rem 0;
        }

        .ll-tagline {
            font-family: 'Orbitron', monospace;
            font-size: 1.6rem;
            font-weight: 900;
            color: #e2eef5;
            line-height: 1.25;
            letter-spacing: 0.03em;
            margin-bottom: 1rem;
        }

        .ll-tagline span { color: var(--cyan); }

        .ll-desc {
            font-size: 0.8rem;
            color: var(--text2);
            line-height: 1.75;
            max-width: 300px;
        }

        .ll-stats {
            display: flex;
            gap: 1rem;
            margin-top: 1.5rem;
        }

        .ll-stat {
            background: rgba(0,212,255,0.05);
            border: 1px solid rgba(0,212,255,0.12);
            border-radius: 8px;
            padding: 0.6rem 0.9rem;
            flex: 1;
        }

        .ll-stat-num {
            font-family: 'Orbitron', monospace;
            font-size: 1.1rem;
            font-weight: 700;
            color: var(--cyan);
        }

        .ll-stat-label {
            font-family: 'Space Mono', monospace;
            font-size: 0.5rem;
            color: var(--text3);
            letter-spacing: 0.15em;
            text-transform: uppercase;
            margin-top: 0.15rem;
        }

        .ll-footer {
            font-family: 'Space Mono', monospace;
            font-size: 0.5rem;
            color: var(--text3);
            letter-spacing: 0.2em;
        }

        .login-right {
            width: 360px;
            flex-shrink: 0;
            background: rgba(3, 10, 20, 0.97);
            padding: 3rem 2.5rem;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }

        .lr-title {
            font-family: 'Orbitron', monospace;
            font-size: 1rem;
            font-weight: 700;
            color: #e2eef5;
            letter-spacing: 0.08em;
            margin-bottom: 0.4rem;
        }

        .lr-sub {
            font-family: 'Space Mono', monospace;
            font-size: 0.55rem;
            color: var(--text3);
            letter-spacing: 0.2em;
            margin-bottom: 2rem;
        }

        .field-label {
            font-family: 'Space Mono', monospace;
            font-size: 0.52rem;
            color: var(--text2);
            letter-spacing: 0.2em;
            text-transform: uppercase;
            margin-bottom: 0.4rem;
        }

        .field-wrap {
            position: relative;
            margin-bottom: 1rem;
        }

        .inp {
            width: 100%;
            padding: 0.85rem 1rem 0.85rem 0.9rem;
            background: rgba(0,212,255,0.04);
            border: 1px solid var(--border);
            border-radius: 6px;
            color: var(--text);
            font-family: 'Space Mono', monospace;
            font-size: 0.75rem;
            outline: none;
            transition: border-color 0.2s, background 0.2s;
        }
        .inp:focus {
            border-color: rgba(0,212,255,0.5);
            background: rgba(0,212,255,0.07);
        }
        .inp::placeholder { color: var(--text3); }

        .btn-login {
            width: 100%;
            padding: 0.95rem;
            background: var(--cyan);
            border: none;
            border-radius: 6px;
            color: #020c18;
            font-family: 'Orbitron', monospace;
            font-weight: 700;
            font-size: 0.65rem;
            letter-spacing: 0.2em;
            cursor: pointer;
            transition: opacity 0.2s, transform 0.1s, box-shadow 0.2s;
            text-transform: uppercase;
            margin-top: 0.75rem;
        }
        .btn-login:hover {
            opacity: 0.88;
            box-shadow: 0 0 30px rgba(0,212,255,0.35);
        }
        .btn-login:active { transform: scale(0.98); }

        .login-error {
            font-family: 'Space Mono', monospace;
            font-size: 0.58rem;
            color: var(--red);
            margin-top: 0.8rem;
            display: none;
            padding: 0.6rem 0.8rem;
            background: rgba(255,51,102,0.08);
            border: 1px solid rgba(255,51,102,0.2);
            border-radius: 4px;
            text-align: center;
        }

        .divider {
            height: 1px;
            background: linear-gradient(90deg, transparent, var(--border2), transparent);
            margin: 1.5rem 0;
        }

        @media (max-width: 700px) {
            .login-panel {
                flex-direction: column;
                width: 94vw;
                min-height: unset;
                border-radius: 10px;
            }
            .login-left {
                padding: 1.5rem;
                border-right: none;
                border-bottom: 1px solid rgba(0,212,255,0.12);
            }
            .ll-hero { padding: 1rem 0; }
            .ll-tagline { font-size: 1.1rem; }
            .ll-desc { display: none; }
            .ll-stats { gap: 0.5rem; }
            .ll-stat { padding: 0.5rem 0.6rem; }
            .ll-stat-num { font-size: 0.85rem; }
            .ll-footer { display: none; }
            .login-right { width: 100%; padding: 1.5rem; }
            .lr-title { font-size: 0.85rem; }
        }

        @media (max-width: 400px) {
            .ll-logo { gap: 0.5rem; }
            .ll-icon { width: 36px; height: 36px; font-size: 1rem; }
            .ll-brand { font-size: 0.75rem; }
            .ll-stats { flex-wrap: wrap; }
        }

        /* === DASHBOARD === */
        #dash {
            display: none;
            flex-direction: column;
            min-height: 100vh;
            position: relative;
            z-index: 1;
        }

        .topbar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0.85rem 1.5rem;
            background: rgba(4,11,20,0.95);
            border-bottom: 1px solid var(--border);
            backdrop-filter: blur(10px);
            position: sticky;
            top: 0;
            z-index: 100;
        }

        .topbar-brand {
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }

        .brand-icon {
            width: 34px;
            height: 34px;
            background: linear-gradient(135deg, var(--cyan), var(--cyan2));
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1rem;
            box-shadow: 0 0 15px rgba(0,212,255,0.3);
        }

        .brand-name {
            font-family: 'Orbitron', monospace;
            font-weight: 900;
            font-size: 1rem;
            color: var(--cyan);
            letter-spacing: 0.1em;
            text-shadow: 0 0 15px rgba(0,212,255,0.4);
        }

        .brand-sub {
            font-family: 'Space Mono', monospace;
            font-size: 0.45rem;
            color: var(--text3);
            letter-spacing: 0.2em;
        }

        .stats-bar {
            display: flex;
            gap: 0.5rem;
            align-items: center;
            flex-wrap: wrap;
        }

        .stat-chip {
            display: flex;
            align-items: center;
            gap: 0.4rem;
            background: var(--bg2);
            border: 1px solid var(--border);
            border-radius: 20px;
            padding: 0.35rem 0.8rem;
            font-family: 'Space Mono', monospace;
            font-size: 0.6rem;
            color: var(--text2);
            transition: border-color 0.3s;
        }
        .stat-chip.active { border-color: var(--border2); }
        .stat-chip .val { color: var(--cyan); font-weight: 700; }

        .btn-logout {
            background: transparent;
            border: 1px solid var(--border);
            border-radius: 4px;
            color: var(--text2);
            font-family: 'Space Mono', monospace;
            font-size: 0.55rem;
            padding: 0.4rem 0.8rem;
            cursor: pointer;
            transition: all 0.2s;
            letter-spacing: 0.1em;
        }
        .btn-logout:hover { border-color: var(--red); color: var(--red); }

        .progress-track {
            height: 2px;
            background: var(--bg2);
            overflow: hidden;
            display: none;
        }
        .progress-fill {
            height: 100%;
            width: 40%;
            background: linear-gradient(90deg, transparent, var(--cyan), transparent);
            animation: scan 1.5s ease-in-out infinite;
        }
        @keyframes scan {
            0%   { transform: translateX(-200%); }
            100% { transform: translateX(400%); }
        }

        .main-grid {
            display: grid;
            grid-template-columns: 340px 1fr;
            gap: 1.25rem;
            padding: 1.25rem;
            flex: 1;
        }

        .left-panel {
            display: flex;
            flex-direction: column;
            gap: 1rem;
            overflow-y: auto;
            max-height: calc(100vh - 80px);
            padding-right: 2px;
        }

        .panel-card {
            background: var(--bg2);
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 1.1rem;
            transition: border-color 0.2s;
        }
        .panel-card:hover { border-color: var(--border2); }

        .card-label {
            font-family: 'Space Mono', monospace;
            font-size: 0.5rem;
            color: var(--text3);
            letter-spacing: 0.3em;
            text-transform: uppercase;
            margin-bottom: 0.8rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        .card-label::after {
            content: '';
            flex: 1;
            height: 1px;
            background: var(--border);
        }

        .hint {
            font-size: 0.7rem;
            color: var(--text3);
            line-height: 1.7;
            margin-bottom: 0.7rem;
        }

        textarea#inp {
            width: 100%;
            height: 220px;
            background: var(--bg);
            border: 1px solid var(--border);
            border-radius: 4px;
            color: #4ade80;
            font-family: 'Space Mono', monospace;
            font-size: 0.65rem;
            padding: 0.8rem;
            resize: vertical;
            outline: none;
            line-height: 1.8;
            transition: border-color 0.2s;
        }
        textarea#inp:focus { border-color: #4ade8040; }
        textarea#inp::placeholder { color: var(--text3); }

        .btn-run {
            width: 100%;
            padding: 0.85rem;
            background: transparent;
            border: 1px solid var(--cyan);
            border-radius: 4px;
            color: var(--cyan);
            font-family: 'Orbitron', monospace;
            font-weight: 700;
            font-size: 0.65rem;
            letter-spacing: 0.2em;
            cursor: pointer;
            transition: all 0.25s;
            text-transform: uppercase;
            position: relative;
            overflow: hidden;
            margin-top: 0.7rem;
        }
        .btn-run::before {
            content: '';
            position: absolute;
            inset: 0;
            background: var(--cyan);
            transform: scaleX(0);
            transform-origin: left;
            transition: transform 0.25s;
            z-index: -1;
        }
        .btn-run:hover::before { transform: scaleX(1); }
        .btn-run:hover { color: #000; box-shadow: 0 0 20px rgba(0,212,255,0.3); }
        .btn-run:disabled {
            border-color: var(--border);
            color: var(--text3);
            cursor: not-allowed;
        }
        .btn-run:disabled::before { display: none; }

        .config-row {
            display: flex;
            align-items: center;
            gap: 0.6rem;
            margin-bottom: 0.6rem;
            font-size: 0.7rem;
        }
        .config-name { color: var(--text2); flex: 1; }
        input[type=range] {
            -webkit-appearance: none;
            width: 90px;
            height: 4px;
            background: var(--border2);
            border-radius: 2px;
            outline: none;
        }
        input[type=range]::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 14px;
            height: 14px;
            border-radius: 50%;
            background: var(--cyan);
            cursor: pointer;
            box-shadow: 0 0 8px rgba(0,212,255,0.5);
        }
        input[type=number] {
            background: var(--bg);
            border: 1px solid var(--border);
            border-radius: 4px;
            color: var(--cyan);
            font-family: 'Space Mono', monospace;
            font-size: 0.6rem;
            padding: 0.25rem 0.5rem;
            width: 52px;
            outline: none;
            text-align: center;
        }
        input[type=number]:focus { border-color: var(--border2); }

        .fleet-section-title {
            font-family: 'Space Mono', monospace;
            font-size: 0.5rem;
            color: var(--text3);
            letter-spacing: 0.3em;
            text-transform: uppercase;
            margin-bottom: 0.6rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        .fleet-section-title::after {
            content: '';
            flex: 1;
            height: 1px;
            background: var(--border);
        }

        .fleet-card {
            background: var(--bg2);
            border: 1px solid var(--border);
            border-left: 3px solid;
            border-radius: 6px;
            padding: 0.9rem;
            margin-bottom: 0.5rem;
            transition: transform 0.15s, border-color 0.15s;
            animation: slide-in 0.3s ease both;
        }
        @keyframes slide-in {
            from { opacity: 0; transform: translateX(-8px); }
            to   { opacity: 1; transform: translateX(0); }
        }
        .fleet-card:hover { transform: translateX(3px); }

        .fleet-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 0.5rem;
        }

        .fleet-unit {
            font-family: 'Space Mono', monospace;
            font-size: 0.5rem;
            letter-spacing: 0.2em;
            color: var(--text3);
            text-transform: uppercase;
        }

        .fleet-count {
            font-family: 'Orbitron', monospace;
            font-size: 1.4rem;
            font-weight: 900;
            color: #f1f5f9;
            line-height: 1;
        }
        .fleet-count span {
            font-family: 'Rajdhani', sans-serif;
            font-size: 0.7rem;
            font-weight: 400;
            color: var(--text2);
        }

        .fleet-km {
            font-size: 0.65rem;
            color: var(--text2);
            margin-top: 0.15rem;
        }

        .btn-maps {
            display: inline-flex;
            align-items: center;
            gap: 0.3rem;
            background: rgba(29, 78, 216, 0.2);
            border: 1px solid #1d4ed8;
            color: #93c5fd;
            padding: 0.45rem 0.75rem;
            border-radius: 4px;
            font-family: 'Space Mono', monospace;
            font-size: 0.55rem;
            text-decoration: none;
            letter-spacing: 0.08em;
            transition: all 0.2s;
            white-space: nowrap;
        }
        .btn-maps:hover {
            background: rgba(29, 78, 216, 0.5);
            box-shadow: 0 0 15px rgba(29,78,216,0.3);
        }

        .stop-list {
            max-height: 140px;
            overflow-y: auto;
            margin-top: 0.5rem;
            border-top: 1px solid var(--border);
            padding-top: 0.4rem;
        }
        .stop-item {
            font-family: 'Space Mono', monospace;
            font-size: 0.55rem;
            color: var(--text2);
            padding: 0.18rem 0;
            border-bottom: 1px solid var(--bg);
            display: flex;
            gap: 0.4rem;
        }
        .stop-item:last-child { border-bottom: none; }
        .stop-num { color: var(--text3); min-width: 18px; }

        .map-container {
            border-radius: 6px;
            border: 1px solid var(--border);
            overflow: hidden;
            min-height: 580px;
            height: calc(100vh - 110px);
            position: relative;
        }
        #map { width: 100%; height: 100%; }
        .leaflet-container { background: var(--bg) !important; }

        .map-badge {
            position: absolute;
            top: 1rem;
            left: 1rem;
            background: rgba(4,11,20,0.9);
            border: 1px solid var(--border2);
            border-radius: 4px;
            padding: 0.5rem 0.8rem;
            font-family: 'Space Mono', monospace;
            font-size: 0.55rem;
            color: var(--cyan);
            letter-spacing: 0.15em;
            z-index: 400;
            pointer-events: none;
        }

        #toast {
            position: fixed;
            bottom: 1.5rem;
            right: 1.5rem;
            background: rgba(127, 29, 29, 0.95);
            border: 1px solid var(--red);
            color: #fca5a5;
            padding: 0.8rem 1.1rem;
            border-radius: 6px;
            font-family: 'Space Mono', monospace;
            font-size: 0.6rem;
            display: none;
            z-index: 9000;
            max-width: 300px;
            line-height: 1.6;
            box-shadow: 0 0 20px rgba(255,51,102,0.2);
            animation: slide-up 0.2s ease;
        }
        @keyframes slide-up {
            from { opacity: 0; transform: translateY(8px); }
            to   { opacity: 1; transform: translateY(0); }
        }

        ::-webkit-scrollbar { width: 4px; height: 4px; }
        ::-webkit-scrollbar-track { background: var(--bg); }
        ::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 2px; }

        @media (max-width: 768px) {
            .main-grid {
                grid-template-columns: 1fr;
                padding: 0.75rem;
                gap: 0.75rem;
            }
            .left-panel { max-height: unset; overflow-y: visible; }
            .map-container { min-height: 340px; height: 55vw; max-height: 420px; }
            .topbar { padding: 0.65rem 1rem; flex-wrap: wrap; gap: 0.5rem; }
            .brand-name { font-size: 0.75rem; }
            .brand-sub  { display: none; }
            .stats-bar { gap: 0.3rem; }
            .stat-chip { padding: 0.25rem 0.55rem; font-size: 0.5rem; }
            textarea#inp { height: 160px; }
        }

        @media (max-width: 480px) {
            .topbar { justify-content: space-between; }
            .stat-chip:nth-child(3) { display: none; }
            .map-container { height: 280px; min-height: 280px; }
        }
    </style>
</head>
<body>

<!-- ===== LOGIN ===== -->
<div id="login">
    <canvas id="login-canvas"></canvas>

    <div class="login-panel">
        <div class="login-left">
            <div class="ll-logo">
                <div class="ll-icon">🗺</div>
                <div>
                    <div class="ll-brand">Y&amp;C ROUTE<br>OPTIMIZER</div>
                    <div class="ll-brand-sub">SISTEMA DE DESPACHO · v2.0</div>
                </div>
            </div>

            <div class="ll-hero">
                <div class="ll-tagline">
                    Rutas <span>inteligentes</span><br>para cada entrega
                </div>
                <div class="ll-desc">
                    Optimización avanzada con OR-Tools + rutas reales
                    por calles vía OSRM. Planifique flotas completas
                    en segundos, directamente desde datos de Excel.
                </div>
                <div class="ll-stats">
                    <div class="ll-stat">
                        <div class="ll-stat-num">200+</div>
                        <div class="ll-stat-label">Paradas</div>
                    </div>
                    <div class="ll-stat">
                        <div class="ll-stat-num">10s</div>
                        <div class="ll-stat-label">Optimización</div>
                    </div>
                    <div class="ll-stat">
                        <div class="ll-stat-num">OSRM</div>
                        <div class="ll-stat-label">Calles reales</div>
                    </div>
                </div>
            </div>

            <div class="ll-footer">© 2026 Y&amp;C SECURE SYSTEMS · PALMARES, CR</div>
        </div>

        <div class="login-right">
            <div class="lr-title">Iniciar sesión</div>
            <div class="lr-sub">ACCESO AL SISTEMA DE RUTAS</div>

            <div class="field-wrap">
                <div class="field-label">Usuario</div>
                <input id="u" class="inp" type="text" placeholder="Ingrese su usuario" autocomplete="off">
            </div>
            <div class="field-wrap">
                <div class="field-label">Contraseña</div>
                <input id="p" class="inp" type="password" placeholder="••••••••">
            </div>

            <button class="btn-login" onclick="auth()">▶ INGRESAR</button>
            <div class="login-error" id="login-err">⚠ Credenciales inválidas. Intente de nuevo.</div>

            <div class="divider"></div>
            <div style="font-family:'Space Mono',monospace;font-size:0.5rem;color:var(--text3);line-height:1.8;text-align:center;">
                Optimización de rutas · Flota múltiple · Calles reales<br>
                Powered by OR-Tools + OSRM + OpenStreetMap
            </div>
        </div>
    </div>
</div>

<!-- ===== TOAST ===== -->
<div id="toast"></div>

<!-- ===== DASHBOARD ===== -->
<div id="dash">

    <div class="topbar">
        <div class="topbar-brand">
            <div class="brand-icon">🗺</div>
            <div>
                <div class="brand-name">Y&amp;C ROUTE OPTIMIZER</div>
                <div class="brand-sub">SISTEMA DE DESPACHO MASIVO · PALMARES, CR</div>
            </div>
        </div>
        <div class="stats-bar">
            <div class="stat-chip" id="sc-stops">
                <span>PARADAS</span><span class="val" id="stat-stops">—</span>
            </div>
            <div class="stat-chip" id="sc-routes">
                <span>RUTAS</span><span class="val" id="stat-routes">—</span>
            </div>
            <div class="stat-chip" id="sc-km">
                <span>KM EST.</span><span class="val" id="stat-km">—</span>
            </div>
            <button class="btn-logout" onclick="logout()">✕ SALIR</button>
        </div>
    </div>

    <div class="progress-track" id="progress-track">
        <div class="progress-fill"></div>
    </div>

    <div class="main-grid">

        <div class="left-panel">

            <div class="panel-card">
                <div class="card-label">Datos de entrega</div>
                <div class="hint">Pegue datos de Excel (Lat, Lon, Nombre) separados por coma, punto y coma, o tabulación.</div>
                <textarea id="inp" placeholder="10.0850, -84.4500, Cliente Uno
10.0723, -84.4310, Cliente Dos
10.0941, -84.4650, Cliente Tres

Desde Excel (tab):
10.0850&#9;-84.4500&#9;Cliente Uno"></textarea>
                <button class="btn-run" id="btn-run" onclick="run()">⚡ OPTIMIZAR RUTAS</button>
            </div>

            <div class="panel-card">
                <div class="card-label">Configuración</div>
                <div class="config-row">
                    <span class="config-name">Vehículos</span>
                    <input type="range" id="cfg-v" min="1" max="10" value="5"
                        oninput="document.getElementById('cfg-v-n').value=this.value">
                    <input type="number" id="cfg-v-n" min="1" max="10" value="5"
                        oninput="document.getElementById('cfg-v').value=this.value">
                </div>
                <div class="config-row">
                    <span class="config-name">Cap. por vehículo</span>
                    <input type="range" id="cfg-c" min="1" max="50" value="18"
                        oninput="document.getElementById('cfg-c-n').value=this.value">
                    <input type="number" id="cfg-c-n" min="1" max="50" value="18"
                        oninput="document.getElementById('cfg-c').value=this.value">
                </div>
            </div>

            <div id="fleet-wrap" style="display:none;">
                <div class="fleet-section-title">Flota asignada</div>
                <div id="fleet"></div>
            </div>

        </div>

        <div class="map-container">
            <div class="map-badge" id="map-badge">BASE · PALMARES</div>
            <div id="map"></div>
        </div>

    </div>
</div>

<script>
let tk = "";
let mapObj = null;
let layers = [];

const COLORS = [
    '#00d4ff', '#ff3366', '#00ff9d', '#ffe033',
    '#bd5fff', '#ff8c42', '#00ffff', '#ff69b4',
    '#7fff00', '#ff4500'
];

// ===== ANIMATED LOGIN CANVAS =====
(function() {
    const canvas = document.getElementById('login-canvas');
    const ctx = canvas.getContext('2d');

    function resize() {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
    }
    resize();
    window.addEventListener('resize', resize);

    const NODE_COUNT = 28;
    const nodes = Array.from({length: NODE_COUNT}, (_, i) => ({
        x: 0.05 + Math.random() * 0.9,
        y: 0.05 + Math.random() * 0.9,
        r: i === 0 ? 7 : (2 + Math.random() * 3),
        isDepot: i === 0,
        pulse: Math.random() * Math.PI * 2,
        pulseSpeed: 0.02 + Math.random() * 0.02
    }));

    const ROUTE_COLORS = ['#00d4ff', '#00ff9d', '#bd5fff', '#ff8c42', '#ff3366'];
    const routes = [];
    const shuffled = [...nodes.slice(1)];
    for (let i = shuffled.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
    }
    const chunkSize = Math.ceil(shuffled.length / 4);
    for (let r = 0; r < 4; r++) {
        const chunk = shuffled.slice(r * chunkSize, (r + 1) * chunkSize);
        routes.push({
            color: ROUTE_COLORS[r],
            stops: [nodes[0], ...chunk, nodes[0]],
            progress: r * 0.25,
            speed: 0.0015 + Math.random() * 0.001
        });
    }

    const trucks = routes.map(route => ({
        route,
        t: route.progress,
        trail: []
    }));

    function getPos(stops, t) {
        const total = stops.length - 1;
        const seg = t * total;
        const i = Math.min(Math.floor(seg), total - 1);
        const frac = seg - i;
        const a = stops[i], b = stops[i + 1];
        return { x: a.x + (b.x - a.x) * frac, y: a.y + (b.y - a.y) * frac };
    }

    function draw(ts) {
        const W = canvas.width, H = canvas.height;
        ctx.clearRect(0, 0, W, H);
        ctx.fillStyle = '#020c18';
        ctx.fillRect(0, 0, W, H);

        ctx.strokeStyle = 'rgba(0,212,255,0.04)';
        ctx.lineWidth = 1;
        const grid = 60;
        for (let x = 0; x < W; x += grid) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke(); }
        for (let y = 0; y < H; y += grid) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke(); }

        routes.forEach(rt => {
            ctx.setLineDash([6, 10]);
            ctx.strokeStyle = rt.color + '28';
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            rt.stops.forEach((s, i) => {
                const px = s.x * W, py = s.y * H;
                i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
            });
            ctx.stroke();
            ctx.setLineDash([]);
        });

        nodes.forEach(n => {
            n.pulse += n.pulseSpeed;
            const px = n.x * W, py = n.y * H;
            if (n.isDepot) {
                [40, 25].forEach((rad, i) => {
                    const alpha = 0.12 - i * 0.04 + Math.sin(n.pulse) * 0.05;
                    ctx.beginPath();
                    ctx.arc(px, py, rad + Math.sin(n.pulse) * 4, 0, Math.PI * 2);
                    ctx.strokeStyle = `rgba(0,212,255,${alpha})`;
                    ctx.lineWidth = 1;
                    ctx.stroke();
                });
                ctx.beginPath();
                ctx.arc(px, py, 8, 0, Math.PI * 2);
                ctx.fillStyle = '#00d4ff';
                ctx.fill();
                ctx.beginPath();
                ctx.arc(px, py, 4, 0, Math.PI * 2);
                ctx.fillStyle = '#020c18';
                ctx.fill();
            } else {
                const alpha = 0.5 + Math.sin(n.pulse) * 0.2;
                ctx.beginPath();
                ctx.arc(px, py, n.r + 1.5, 0, Math.PI * 2);
                ctx.fillStyle = `rgba(0,212,255,${alpha * 0.15})`;
                ctx.fill();
                ctx.beginPath();
                ctx.arc(px, py, n.r, 0, Math.PI * 2);
                ctx.fillStyle = `rgba(0,212,255,${alpha * 0.7})`;
                ctx.fill();
            }
        });

        trucks.forEach(truck => {
            truck.t = (truck.t + truck.route.speed) % 1;
            const pos = getPos(truck.route.stops, truck.t);
            const px = pos.x * W, py = pos.y * H;
            truck.trail.push({x: px, y: py, age: 0});
            if (truck.trail.length > 30) truck.trail.shift();
            truck.trail.forEach((pt, i) => {
                pt.age++;
                const alpha = (1 - pt.age / 32) * 0.6;
                ctx.beginPath();
                ctx.arc(pt.x, pt.y, 2.5 * (1 - pt.age / 32), 0, Math.PI * 2);
                ctx.fillStyle = truck.route.color + Math.round(alpha * 255).toString(16).padStart(2, '0');
                ctx.fill();
            });
            ctx.beginPath();
            ctx.arc(px, py, 5, 0, Math.PI * 2);
            ctx.fillStyle = truck.route.color;
            ctx.fill();
            ctx.beginPath();
            ctx.arc(px, py, 2.5, 0, Math.PI * 2);
            ctx.fillStyle = '#020c18';
            ctx.fill();
        });

        requestAnimationFrame(draw);
    }

    requestAnimationFrame(draw);
})();

// === MAP ===
function initMap() {
    mapObj = L.map('map', { zoomControl: true, preferCanvas: true }).setView([10.0605, -84.4372], 11);
    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
        attribution: '© OpenStreetMap © Carto',
        maxZoom: 19
    }).addTo(mapObj);

    const depotHTML = `
        <div style="width:18px;height:18px;background:#00d4ff;border-radius:50%;
            border:3px solid rgba(255,255,255,0.9);
            box-shadow:0 0 15px rgba(0,212,255,0.8), 0 0 30px rgba(0,212,255,0.3);">
        </div>`;
    L.marker([10.0605, -84.4372], {
        icon: L.divIcon({ html: depotHTML, className: '', iconAnchor: [9, 9] })
    }).addTo(mapObj).bindPopup('<div style="font-family:monospace;font-size:11px;"><b>🏭 BASE PALMARES</b></div>');
}

// === AUTH ===
async function auth() {
    const u = document.getElementById('u').value.trim();
    const p = document.getElementById('p').value;
    const err = document.getElementById('login-err');
    err.style.display = 'none';

    if (!u || !p) {
        err.style.display = 'block';
        err.textContent = '⚠ Ingrese usuario y contraseña.';
        return;
    }

    const fd = new FormData();
    fd.append('username', u);
    fd.append('password', p);

    try {
        const r = await fetch('/token', { method: 'POST', body: fd });
        if (r.ok) {
            tk = (await r.json()).access_token;
            document.getElementById('login').style.display = 'none';
            document.getElementById('dash').style.display = 'flex';
            setTimeout(() => { initMap(); mapObj && mapObj.invalidateSize(); }, 250);
        } else {
            err.style.display = 'block';
            err.textContent = '⚠ Credenciales inválidas.';
        }
    } catch (e) {
        err.style.display = 'block';
        err.textContent = '⚠ Error de conexión con el servidor.';
    }
}

function logout() {
    tk = "";
    document.getElementById('dash').style.display = 'none';
    document.getElementById('login').style.display = 'flex';
    document.getElementById('login-err').style.display = 'none';
    document.getElementById('u').value = '';
    document.getElementById('p').value = '';
    ['stat-stops','stat-routes','stat-km'].forEach(id => document.getElementById(id).textContent = '—');
    document.getElementById('fleet').innerHTML = '';
    document.getElementById('fleet-wrap').style.display = 'none';
}

// === PARSE ===
function parseInput(raw) {
    const deliveries = [];
    const lines = raw.split('\n').map(l => l.trim()).filter(l => l);

    for (let i = 0; i < lines.length; i++) {
        const line = lines[i];
        if (line.startsWith('#') || line.startsWith('//')) continue;

        let parts;
        if (line.includes('\t'))       parts = line.split('\t').map(p => p.trim()).filter(p => p);
        else if (line.includes(';'))   parts = line.split(';').map(p => p.trim()).filter(p => p);
        else                           parts = line.split(',').map(p => p.trim()).filter(p => p);

        if (parts.length < 2) continue;

        const a = parseFloat(parts[0].replace(',', '.'));
        const b = parseFloat(parts[1].replace(',', '.'));
        if (isNaN(a) || isNaN(b)) continue;

        let lat = a, lon = b;
        if (b >= 8 && b <= 11.5 && a >= -86 && a <= -82) { lat = b; lon = a; }

        const nombre = parts.slice(2).join(' ').trim() || `Parada ${deliveries.length + 1}`;
        deliveries.push({ id: i, lat, lon, descripcion: nombre });
    }
    return deliveries;
}

// === GOOGLE MAPS URL — abre navegación paso a paso ===
// Usa el formato "dirección múltiple" que funciona en móvil y desktop
function buildMapsUrl(depot, waypoints) {
    // Google Maps URL con múltiples destinos encadenados (no necesita API key)
    // Formato: /maps/dir/origen/wp1/wp2/.../destino_final
    // Límite práctico: ~25 puntos totales

    const all = [depot, ...waypoints.slice(0, 23), depot];
    const parts = all.map(c => `${c[0]},${c[1]}`);
    return "https://www.google.com/maps/dir/" + parts.join('/');
}

// === OPTIMIZE ===
async function run() {
    const raw = document.getElementById('inp').value;
    const deliveries = parseInput(raw);

    if (deliveries.length === 0) {
        showToast('⚠ No se encontraron coordenadas válidas.\nFormato: Lat, Lon, Nombre');
        return;
    }
    if (deliveries.length > 200) {
        showToast('⚠ Máximo 200 paradas por optimización.');
        return;
    }
    if (!tk) {
        showToast('⚠ Sesión expirada. Por favor inicie sesión de nuevo.');
        logout();
        return;
    }

    const nv = parseInt(document.getElementById('cfg-v-n').value) || 5;
    const cap = parseInt(document.getElementById('cfg-c-n').value) || 18;

    setLoading(true);
    try {
        const r = await fetch('/optimize', {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${tk}`,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ deliveries, num_vehicles: nv, capacity_per_vehicle: cap })
        });

        if (r.status === 401) {
            showToast('⚠ Sesión expirada. Vuelva a iniciar sesión.');
            logout();
            return;
        }
        if (!r.ok) {
            const err = await r.json();
            throw new Error(err.detail || 'Error del servidor');
        }

        const data = await r.json();
        renderRoutes(data, deliveries);
    } catch (e) {
        showToast('❌ Error: ' + e.message);
    } finally {
        setLoading(false);
    }
}

// === RENDER ===
function renderRoutes(data, deliveries) {
    layers.forEach(l => mapObj.removeLayer(l));
    layers = [];
    document.getElementById('fleet').innerHTML = '';

    if (!data.routes || data.routes.length === 0) {
        showToast('⚠ No se generaron rutas. Verifique los datos.');
        return;
    }

    let totalKm = 0;
    const depot = [10.0605, -84.4372];

    data.routes.forEach((rt, i) => {
        const color = COLORS[i % COLORS.length];
        totalKm += rt.km || 0;

        const poly = L.polyline(rt.route, {
            color,
            weight: 2.5,
            opacity: 0.8
        }).addTo(mapObj);
        layers.push(poly);

        (rt.stop_info || []).forEach((stop, j) => {
            const icon = L.divIcon({
                html: `<div style="
                    background:${color};width:22px;height:22px;border-radius:50%;
                    border:2px solid rgba(0,0,0,0.6);
                    display:flex;align-items:center;justify-content:center;
                    font-family:monospace;font-size:8px;font-weight:700;color:#000;
                    box-shadow:0 0 8px ${color}88;">
                    ${j+1}
                </div>`,
                className: '',
                iconAnchor: [11, 11]
            });

            const m = L.marker([stop.lat, stop.lon], { icon })
                .addTo(mapObj)
                .bindPopup(`
                    <div style="font-family:monospace;font-size:11px;line-height:1.6;">
                        <b style="color:${color}">Unidad ${rt.vehicle} — Parada ${j+1}</b><br>
                        ${stop.nombre}<br>
                        <small style="color:#888">${stop.lat.toFixed(5)}, ${stop.lon.toFixed(5)}</small>
                    </div>
                `);
            layers.push(m);
        });

        // Construir URL de Google Maps desde el frontend con el formato /dir/
        const waypoints = (rt.stop_info || []).map(s => [s.lat, s.lon]);
        const mapsUrl = buildMapsUrl(depot, waypoints);

        const card = document.createElement('div');
        card.className = 'fleet-card';
        card.style.borderLeftColor = color;
        card.style.animationDelay = `${i * 0.06}s`;
        card.innerHTML = `
            <div class="fleet-header">
                <div>
                    <div class="fleet-unit">UNIDAD ${rt.vehicle}</div>
                    <div class="fleet-count">${rt.stops} <span>paradas</span></div>
                    <div class="fleet-km">
                        ${rt.km} km
                        <span style="font-size:0.5rem;padding:0.1rem 0.35rem;border-radius:3px;margin-left:0.3rem;
                            background:${rt.km_source==='osrm' ? 'rgba(0,255,157,0.12)' : 'rgba(255,224,51,0.1)'};
                            color:${rt.km_source==='osrm' ? 'var(--green)' : 'var(--yellow)'};
                            border:1px solid ${rt.km_source==='osrm' ? 'rgba(0,255,157,0.3)' : 'rgba(255,224,51,0.3)'};">
                            ${rt.km_source==='osrm' ? '● OSRM real' : '~ estimado'}
                        </span>
                    </div>
                </div>
                <a href="${mapsUrl}" target="_blank" rel="noopener" class="btn-maps">🗺 Navegar</a>
            </div>
            ${rt.stop_info && rt.stop_info.length > 0 ? `
            <div class="stop-list">
                ${rt.stop_info.map((s, j) => `
                    <div class="stop-item">
                        <span class="stop-num">${j + 1}.</span>
                        <span>${s.nombre}</span>
                    </div>
                `).join('')}
            </div>` : ''}`;
        document.getElementById('fleet').appendChild(card);
    });

    document.getElementById('stat-stops').textContent = deliveries.length;
    document.getElementById('stat-routes').textContent = data.routes.length;
    document.getElementById('stat-km').textContent = totalKm.toFixed(0);
    ['sc-stops','sc-routes','sc-km'].forEach(id => {
        document.getElementById(id).classList.add('active');
    });

    document.getElementById('fleet-wrap').style.display = 'block';

    const allCoords = data.routes.flatMap(r => r.route);
    if (allCoords.length > 0) {
        mapObj.fitBounds(L.latLngBounds(allCoords), { padding: [30, 30] });
    }

    document.getElementById('map-badge').textContent =
        `BASE PALMARES · ${data.routes.length} RUTAS · ${deliveries.length} PARADAS`;
}

// === HELPERS ===
function setLoading(on) {
    document.getElementById('progress-track').style.display = on ? 'block' : 'none';
    const btn = document.getElementById('btn-run');
    btn.disabled = on;
    btn.textContent = on ? '⏳ OPTIMIZANDO...' : '⚡ OPTIMIZAR RUTAS';
}

let toastTimer;
function showToast(msg) {
    clearTimeout(toastTimer);
    const t = document.getElementById('toast');
    t.textContent = msg;
    t.style.display = 'block';
    toastTimer = setTimeout(() => { t.style.display = 'none'; }, 5000);
}

document.addEventListener('keydown', e => {
    if (e.key === 'Enter' && document.getElementById('login').style.display !== 'none') auth();
});
</script>
</body>
</html>
"""

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)