# ana_pluviometry.py
# Este módulo lida com a obtenção e manipulação de dados pluviométricos da ANA (Agência Nacional de Águas)
# Inclui funções para buscar metadados de estações e séries históricas de precipitação
from datetime import datetime, timedelta
from typing import List, Optional, Tuple
import requests
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, box
import xarray as xr
from math import radians, sin, cos, sqrt, atan2

ANA_BASE = "https://apitempo.inmet.gov.br"        # exemplo público
ANA_STATIONS = "/stations"                        # ↩ troque pelo endpoint real da ANA
ANA_SERIES   = "/serie"                           # ↩ idem
ANA_USER = "07311315336"
ANA_PASS = "13gsu43k"

# Função auxiliar para converter limites em string bbox
# bounds: (xmin, ymin, xmax, ymax)
def _to_bbox(bounds: Tuple[float,float,float,float]) -> str:
    xmin, ymin, xmax, ymax = bounds
    return f"{xmin},{ymin},{xmax},{ymax}"

def login_ana(user: str, password: str) -> requests.Session:
    """Realiza login na plataforma da ANA/Hidroweb e retorna uma sessão autenticada."""
    login_url = "https://www.snirh.gov.br/hidroweb/rest/auth/login"
    payload = {"login": user, "senha": password}
    session = requests.Session()
    resp = session.post(login_url, json=payload, timeout=30)
    resp.raise_for_status()
    return session

def obter_token_ana(identificador: str, senha: str) -> str:
    """Obtém o token de autenticação da API HIDRO_Webservice da ANA."""
    url = "https://www.ana.gov.br/hidrowebservice/EstacoesTelemetricas/OAUth/v1"
    headers = {
        "Identificador": identificador,
        "Senha": senha
    }
    try:
        response = requests.get(url, headers=headers)
        print(response.status_code)
        print(response.text)
        response.raise_for_status()
        data = response.json()
        # O token está dentro de items["tokenautenticacao"]
        token = data.get("items", {}).get("tokenautenticacao")
        if not token:
            raise ValueError("Token de autenticação não recebido")
        return token
    except Exception as e:
        print(f"Erro ao obter token de autenticação: {e}")
        print(f"Resposta: {getattr(e, 'response', None)}")
        raise

def criar_sessao_autenticada(identificador: str, senha: str) -> requests.Session:
    """Cria e retorna uma sessão autenticada para a API da ANA."""
    token = obter_token_ana(identificador, senha)
    session = requests.Session()
    session.headers.update({
        "Authorization": f"Bearer {token}",
        "Accept": "application/json"
    })
    return session

def buscar_inventario_estacoes(token: str) -> list:
    """Busca o inventário completo de estações da ANA."""
    url = "https://www.ana.gov.br/hidrowebservice/EstacoesTelemetricas/HidroInventarioEstacoes/v1"
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(url, headers=headers)
    print(response.status_code)
    print(response.text)
    response.raise_for_status()
    return response.json()["items"]

def dentro_bounding_box(lat, lon, box):
    lat_min, lat_max, lon_min, lon_max = box
    return lat_min <= lat <= lat_max and lon_min <= lon <= lon_max

def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # km
    dlat = radians(lat2 - lat1)
    dlon = radians(lat2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))

def filtrar_estacoes_geograficamente(estacoes, bounding_box=None, centro_busca=None, raio_km=None):
    # Calcula o centro do bounding box
    if bounding_box:
        lat_min, lat_max, lon_min, lon_max = bounding_box
        centro_busca = ((lat_min + lat_max)/2, (lon_min + lon_max)/2)
    # Filtra estações dentro do raio especificado
    if centro_busca and raio_km:
        filtradas = [est for est in estacoes if haversine(est['latitude'], est['longitude'], *centro_busca) <= raio_km]
        return filtradas
    return estacoes
    """Filtra estações por bounding box e/ou por raio (em km) a partir de um centro."""
    filtradas = estacoes
    if bounding_box:
        filtradas = [est for est in filtradas if dentro_bounding_box(est["latitude"], est["longitude"], bounding_box)]
    if centro_busca & raio_km:
        filtradas = [est for est in filtradas if haversine(est["latitude"], est["longitude"], *centro_busca) <= raio_km]
    return filtradas

def buscar_serie_estacao(token, codigo_estacao, data_inicial, data_final=None, intervalo="DIAS_30"):
    """
    Busca série histórica de precipitação para uma estação, enviando uma data por vez no parâmetro 'Data de Busca (yyyy-MM-dd)'.
    """
    import pandas as pd
    from datetime import datetime, timedelta
    from urllib.parse import quote_plus
    url = "https://www.ana.gov.br/hidrowebservice/EstacoesTelemetricas/HidroinfoanaSerieTelemetricaAdotada/v1"
    headers = {"Authorization": f"Bearer {token}"}
    # Gera lista de datas entre data_inicial e data_final (inclusive)
    datas = []
    data_ini_dt = datetime.strptime(data_inicial, "%Y-%m-%d")
    if data_final:
        data_fim_dt = datetime.strptime(data_final, "%Y-%m-%d")
    else:
        data_fim_dt = data_ini_dt
    delta = data_fim_dt - data_ini_dt
    for i in range(delta.days + 1):
        datas.append((data_ini_dt + timedelta(days=i)).strftime("%Y-%m-%d"))
    resultados = []
    for data in datas:
        query_params = [
            ("Código da Estação", codigo_estacao),
            ("Tipo Filtro Data", "DATA_LEITURA"),
            ("Data de Busca (yyyy-MM-dd)", data),
            ("Range Intervalo de busca", intervalo)
        ]
        query_string = "&".join([f"{quote_plus(k)}={quote_plus(str(v))}" for k, v in query_params])
        full_url = url + "?" + query_string
        print("[ANA DEBUG] Série URL:", full_url)
        response = requests.get(full_url, headers=headers)
        print(response.status_code)
        print(response.text)
        response.raise_for_status()
        items = response.json().get("items", [])
        if items:
            resultados.extend([item for item in items if item.get("Chuva_Adotada") is not None])
    return resultados
    url = "https://www.ana.gov.br/hidrowebservice/EstacoesTelemetricas/HidroinfoanaSerieTelemetricaAdotada"
    headers = {"Authorization": f"Bearer {token}"}
    params = {
        "codigoEstacao": codigo_estacao,
        "codigoTipoGrandeza": 1,  # precipitação
        "dataInicial": data_inicial,  # formato yyyy-MM-dd
        "dataFinal": data_final
    }
    response = requests.get(url, headers=headers, params=params)
    print(response.status_code)
    print(response.text)
    response.raise_for_status()
    items = response.json().get("items", [])
    if items:
        return [item for item in items if item.get("Chuva_Adotada") is not None]
    return []

# Exemplo de fluxo principal (pode ser adaptado para scripts ou notebooks):
def fluxo_completo_ana(identificador, senha, bounding_box, centro_busca, raio_km, data_inicial, data_final):
    token = obter_token_ana(identificador, senha)
    estacoes = buscar_inventario_estacoes(token)
    estacoes_filtradas = filtrar_estacoes_geograficamente(estacoes, bounding_box, centro_busca, raio_km)
    series = {}
    for est in estacoes_filtradas:
        codigo = est["codigoEstacao"]
        serie = buscar_serie_estacao(token, codigo, data_inicial, data_final)
        series[codigo] = serie
    return series

# Busca metadados das estações dentro do bbox informado
# Retorna um GeoDataFrame com as estações encontradas
def list_stations(bounds, session, uf=None, data_ini=None, data_fim=None) -> gpd.GeoDataFrame:
    """
    Busca inventário de estações da ANA dentro da bbox + filtros, usando exatamente os parâmetros do curl oficial.
    bounds = (xmin, ymin, xmax, ymax)
    uf = sigla da unidade federativa, ex: "CE"
    datas no formato YYYY-MM-DD
    """
    xmin, ymin, xmax, ymax = bounds
    url = "https://www.ana.gov.br/hidrowebservice/EstacoesTelemetricas/HidroInventarioEstacoes/v1"
    query_params = []
    if data_ini:
        query_params.append(("Data Atualização Inicial (yyyy-MM-dd)", data_ini))
    if data_fim:
        query_params.append(("Data Atualização Final (yyyy-MM-dd)", data_fim))
    if uf:
        query_params.append(("Unidade Federativa", uf))
    # Monta a query string manualmente para preservar nomes e acentuação
    from urllib.parse import urlencode, quote_plus
    query_string = "&".join([f"{quote_plus(k)}={quote_plus(v)}" for k, v in query_params])
    full_url = url + ("?" + query_string if query_string else "")
    headers = {"accept": "*/*"}
    print("[ANA DEBUG] URL:", full_url)
    print("[ANA DEBUG] Parâmetros:", query_params)
    r = session.get(full_url, headers=headers, timeout=60)
    print("[ANA DEBUG] Status:", r.status_code)
    print("[ANA DEBUG] Response:", r.text[:500])  # Mostra só o início para não poluir
    r.raise_for_status()
    items = r.json().get("items", [])
    df = pd.DataFrame(items)
    if df.empty:
        return gpd.GeoDataFrame(columns=["codigoEstacao","nomeEstacao","Latitude","Longitude","geometry"], crs="epsg:4326")
    df["Latitude"]  = df["Latitude"].astype(float)
    df["Longitude"] = df["Longitude"].astype(float)
    gdf = gpd.GeoDataFrame(
        df,
        geometry=[Point(xy) for xy in zip(df["Longitude"], df["Latitude"])],
        crs="epsg:4326"
    )
    return gdf

# Baixa a série horária/diária de precipitação de uma estação específica
# cod: código da estação
# start, end: datas inicial e final no formato 'YYYY-MM-DD'
def fetch_series(cod, start, end, session, tipo_filtro="DATA_LEITURA", intervalo=None) -> pd.Series:
    """
    Busca série temporal detalhada de precipitação para uma estação, usando o endpoint e parâmetros exatos do curl fornecido.
    Retorna pd.Series indexed by datetime.
    """
    url = "https://www.ana.gov.br/hidrowebservice/EstacoesTelemetricas/HidroinfoanaSerieTelemetricaAdotada/v1"
    from urllib.parse import quote_plus
    query_params = [
        ("Código da Estação", cod),
        ("Tipo Filtro Data", tipo_filtro),
        ("Data de Busca (yyyy-MM-dd)", start),
        ("Range Intervalo de busca", intervalo)
    ]
    # Remove o parâmetro de data final para seguir o formato especificado
    query_params = [param for param in query_params if param[0] != "Data de Busca (yyyy-MM-dd)"]
    query_string = "&".join([f"{quote_plus(k)}={quote_plus(v)}" for k, v in query_params])
    headers = {"accept": "*/*"}
    full_url = url + "?" + query_string
    print("[ANA DEBUG] Série URL:", full_url)
    r = session.get(full_url, headers=headers, timeout=60)
    print("[ANA DEBUG] Série Status:", r.status_code)
    print("[ANA DEBUG] Série Response:", r.text[:500])
    r.raise_for_status()
    items = r.json().get("items", [])
    if not items:
        return pd.Series(dtype=float)
    df = pd.DataFrame(items)
    if {"Data", "Valor"}.issubset(df.columns):
        df["Data"] = pd.to_datetime(df["Data"])
        df.set_index("Data", inplace=True)
        return df["Valor"].astype(float)
    elif {"Data", "Chuva_Adotada"}.issubset(df.columns):
        df["Data"] = pd.to_datetime(df["Data"])
        df.set_index("Data", inplace=True)
        return df["Chuva_Adotada"].astype(float)
    return pd.Series(dtype=float)

    if resultados:
        return pd.concat(resultados).sort_index()
    else:
        return pd.Series(dtype=float)

# Une todas as séries de precipitação das estações em um único Dataset xarray
# stations: GeoDataFrame das estações
# start, end: datas inicial e final
def make_dataset(
    stations: gpd.GeoDataFrame,
    start: str,
    end: str,
    session: requests.Session = None,
    range_intervalo: str = "MINUTO_5"
) -> xr.Dataset:
    """Une todas as séries num único Dataset (tempo × estação)."""
    arrays = []
    for _, row in stations.iterrows():
        s = fetch_series(row["codigoestacao"], start, end, session=session, intervalo=range_intervalo)
        arrays.append(s.rename(row["codigoestacao"]))
    if arrays:
        df = pd.concat(arrays, axis=1)
        return df.to_xarray()
    return xr.Dataset()

def obter_chuva_diaria_em_intervalo(estacoes, data_inicio, data_fim, token, intervalo="HORA_24"):
    """
    Busca a chuva diária adotada para cada dia do intervalo e para todas as estações fornecidas.
    Retorna um DataFrame com colunas 'dia' e 'Chuva_Adotada'.
    """
    from datetime import datetime, timedelta
    import pandas as pd
    data_atual = datetime.strptime(data_inicio, "%Y-%m-%d")
    data_final = datetime.strptime(data_fim, "%Y-%m-%d")
    resultados = []
    datas = []
    for i in range((data_final - data_atual).days + 1):
        data_str = (data_atual + timedelta(days=i)).strftime("%Y-%m-%d")
        chuva_dia = []
        for est in estacoes:
            if isinstance(est, dict):
                codigo = est.get("codigoEstacao")
            elif hasattr(est, "codigoEstacao"):
                codigo = est.codigoEstacao
            elif hasattr(est, "__getitem__") and "codigoEstacao" in est:
                codigo = est["codigoEstacao"]
            else:
                continue
            try:
                items = buscar_serie_estacao(token, codigo, data_str, data_str)
                valores = [float(item["Chuva_Adotada"]) for item in items if "Chuva_Adotada" in item and item["Chuva_Adotada"] is not None]
                if valores:
                    chuva_dia.append(sum(valores))
            except Exception:
                continue
        if chuva_dia:
            resultados.append(sum(chuva_dia)/len(chuva_dia))
        else:
            resultados.append(float('nan'))
        datas.append(data_str)
    df_result = pd.DataFrame({"dia": datas, "Chuva_Adotada": resultados})
    return df_result
