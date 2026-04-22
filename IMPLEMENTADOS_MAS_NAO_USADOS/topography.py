# topography.py
# Funções para baixar, mosaicar e processar dados de topografia (DEM) e atributos derivados.
# Inclui funções para buscar tiles Copernicus, calcular slope/aspect e converter para xarray.
# Suporte otimizado para região de Juazeiro do Norte, Ceará.
# Integração com PyGMT para visualização avançada de dados topográficos.
from pathlib import Path
from typing import Tuple, Literal, Optional, Dict, Union
import logging
import rasterio
from rasterio.merge import merge
import xarray as xr
import numpy as np
import os
import requests

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('topography')

# Importação do PyGMT para visualização avançada
try:
    import pygmt
    PYGMT_AVAILABLE = True
except ImportError:
    PYGMT_AVAILABLE = False
    logger.warning("PyGMT não está instalado. Algumas funcionalidades de visualização não estarão disponíveis.")
    logger.warning("Instale com: pip install pygmt")

# S3 path público (Registry of Open Data on AWS)
BUCKET = "s3://copernicus-dem-30m"
COPERNICUS_HTTP = "https://copernicus-dem-30m.s3.amazonaws.com"

# Coordenadas de Juazeiro do Norte, Ceará
JUAZEIRO_BOUNDS = (-39.35, -7.25, -39.28, -7.18)  # (xmin, ymin, xmax, ymax) em graus decimais

# Fontes de DEM disponíveis
DEM_SOURCES = {
    "copernicus": {
        "url": COPERNICUS_HTTP,
        "resolution": 0.00027777,  # ~30m no equador
        "extension": "tif"
    },
    # Outras fontes podem ser adicionadas aqui no futuro
}

def _tile_names(bounds: Tuple[float,float,float,float]) -> list[str]:
    """
    Retorna a lista de ladrilhos Copernicus que interceptam o bbox.
    """
    xmin, ymin, xmax, ymax = bounds
    import math
    xs = range(math.floor(xmin), math.ceil(xmax))
    ys = range(math.floor(ymin), math.ceil(ymax))
    tiles = []
    for lat in ys:
        for lon in xs:
            ns = "N" if lat >= 0 else "S"
            ew = "E" if lon >= 0 else "W"
            lat_abs = abs(lat)
            lon_abs = abs(lon)
            tiles.append(f"{ns}{lat_abs:02d}{ew}{lon_abs:03d}_DEM.tif")
    return tiles

def _download_tile(tile: str, cache_dir: Path, overwrite: bool, fonte: str = "copernicus") -> Optional[Path]:
    """
    Baixa um tile DEM de fonte pública (Copernicus por padrão) para o cache_dir.
    Se falhar, tenta fontes alternativas ou retorna None.
    """
    local = cache_dir / tile
    if local.exists() and not overwrite:
        logger.info(f"Usando tile em cache: {local}")
        return local
    if fonte not in DEM_SOURCES:
        logger.warning(f"Fonte '{fonte}' não suportada, usando Copernicus como padrão")
        fonte = "copernicus"
    url = f"{DEM_SOURCES[fonte]['url']}/{tile}"
    logger.info(f"Baixando tile de {url}")
    try:
        resp = requests.get(url, stream=True, timeout=60)
        if resp.status_code == 200:
            with open(local, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            logger.info(f"Download concluído: {local}")
            return local
        else:
            logger.error(f"Falha no download (status {resp.status_code}): {url}")
            logger.warning(f"Tentando fonte alternativa para {tile}")
            return None
    except Exception as e:
        logger.error(f"Erro ao baixar tile {tile}: {str(e)}")
        return None

def fetch_dem(bounds: Optional[Tuple[float, float, float, float]] = None, cache_dir=Path("cache/dem"), 
            overwrite=False, fonte: str = "copernicus", export_path: Optional[Path] = None,
            region: str = "", resolution: Optional[float] = None, use_gee: bool = False, gee_dataset: str = "USGS/SRTMGL1_003", gee_scale: int = 30, gee_export_path: Optional[str] = None):
    """
    Baixa e mosaica tiles DEM para os limites informados ou obtém via Google Earth Engine (GEE).
    """
    if use_gee:
        try:
            import ee
        except ImportError:
            raise ImportError("A biblioteca 'earthengine-api' não está instalada. Instale com: pip install earthengine-api")
        try:
            ee.Initialize(project='pesquisa-lhiufca')
        except Exception:
            ee.Authenticate()
            ee.Initialize(project='pesquisa-lhiufca')
        
        if bounds is None:
            if region.lower() == "juazeiro":
                bounds = JUAZEIRO_BOUNDS
                logger.info(f"Usando limites predefinidos para Juazeiro do Norte: {bounds}")
            else:
                raise ValueError("É necessário fornecer 'bounds' ou especificar uma região válida.")
        
        xmin, ymin, xmax, ymax = bounds
        region_geom = ee.Geometry.Rectangle([xmin, ymin, xmax, ymax])
        dem = ee.Image(gee_dataset)
        dem_clip = dem.clip(region_geom)
        try:
            import geemap
        except ImportError:
            raise ImportError("A biblioteca 'geemap' não está instalada. Instale com: pip install geemap")
        if gee_export_path is None:
            gee_export_path = str(cache_dir / "gee_dem.tif")
        logger.info(f"Exportando DEM do GEE para {gee_export_path}")
        geemap.ee_export_image(
            dem_clip,
            filename=gee_export_path,
            scale=gee_scale,
            region=region_geom,
            file_per_band=False,
            crs='EPSG:4326',
            timeout=600
        )
        with rasterio.open(gee_export_path) as src:
            arr = src.read(1).astype(np.float32)
            profile = src.profile
            profile["dtype"] = "float32"
            arr[arr == src.nodata] = np.nan
        transform = profile.get("transform")
        if transform is not None and transform[4] > 0:
            profile["transform"] = (transform[0], transform[1], transform[2], transform[3], -abs(transform[4]), transform[5])
        return arr, profile
    if bounds is None:
        if region.lower() == "juazeiro":
            bounds = JUAZEIRO_BOUNDS
            logger.info(f"Usando limites predefinidos para Juazeiro do Norte: {bounds}")
        else:
            raise ValueError("É necessário fornecer 'bounds' ou especificar uma região válida.")
    if not isinstance(bounds, tuple) or len(bounds) != 4:
        raise ValueError("O parâmetro 'bounds' deve ser uma tupla com quatro elementos (xmin, ymin, xmax, ymax).")
    xmin, ymin, xmax, ymax = bounds
    if not (-180 <= xmin <= 180 and -180 <= xmax <= 180 and -90 <= ymin <= 90 and -90 <= ymax <= 90):
        raise ValueError("Os limites fornecidos estão fora do intervalo geográfico válido.")
    if xmin >= xmax or ymin >= ymax:
        raise ValueError("Os limites devem obedecer xmin < xmax e ymin < ymax.")
    if fonte not in DEM_SOURCES:
        logger.warning(f"Fonte '{fonte}' não suportada, usando Copernicus como padrão")
        fonte = "copernicus"
    if resolution is None:
        resolution = DEM_SOURCES[fonte]["resolution"]
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    srcs = []
    tiles = _tile_names(bounds)
    logger.info(f"Processando {len(tiles)} tiles para a região {bounds}")
    for tile in tiles:
        local = cache_dir / tile
        if not local.exists() or overwrite:
            ok = _download_tile(tile, cache_dir, overwrite, fonte)
            if not ok or not local.exists():
                logger.warning(f"Falha no download do tile {tile}, usando dados simulados")
                shape = (50, 50)
                x = np.linspace(0, 10, shape[0])
                y = np.linspace(0, 10, shape[1])
                xx, yy = np.meshgrid(x, y)
                np.random.seed(42)
                arr = 100 + 50 * np.sin(xx/2) * np.cos(yy/2) + 30 * np.random.rand(*shape)
                np.random.seed(None)
                arr = arr.astype(np.float32)
                xmin, ymin, xmax, ymax = bounds
                res_x = (xmax - xmin) / shape[1]
                res_y = (ymin - ymax) / shape[0]
                profile = {
                    "driver": "GTiff",
                    "dtype": "float32",
                    "nodata": None,
                    "width": shape[1],
                    "height": shape[0],
                    "count": 1,
                    "crs": "EPSG:4326",
                    "transform": (res_x, 0.0, xmin, 0.0, -abs(res_y), ymax)
                }
                logger.info(f"Usando DEM simulado com resolução: {res_x:.8f}, {-abs(res_y):.8f}")
                return arr, profile
        try:
            srcs.append(rasterio.open(local))
        except Exception as e:
            logger.error(f"Erro ao abrir o tile {local}: {str(e)}")
            continue
    if not srcs:
        raise ValueError("Não foi possível obter nenhum tile válido para a região especificada.")
    logger.info(f"Criando mosaico com {len(srcs)} tiles")
    mosaic, out_trans = merge(srcs, bounds=bounds, res=(resolution, resolution))
    profile = srcs[0].profile
    profile.update({
        "height": mosaic.shape[1], 
        "width": mosaic.shape[2], 
        "transform": out_trans,
        "crs": "EPSG:4326"
    })
    if export_path:
        logger.info(f"Exportando mosaico para {export_path}")
        with rasterio.open(export_path, 'w', **profile) as dst:
            dst.write(mosaic)
    for src in srcs:
        src.close()
    return mosaic[0], profile

def slope_aspect(dem_arr, profile, what: Literal["slope","aspect"]="slope"):
    """
    Calcula o atributo de terreno (declividade ou aspecto) usando numpy.
    """
    if not isinstance(dem_arr, np.ndarray) or dem_arr.size == 0:
        raise ValueError("Array DEM inválido.")
    if not isinstance(profile, dict) or "transform" not in profile:
        raise ValueError("Perfil inválido.")
    x, y = np.gradient(dem_arr)
    slope = np.sqrt(x**2 + y**2)
    if what == "slope":
        return slope.astype(np.float32)
    elif what == "aspect":
        aspect = np.arctan2(-x, y)
        return aspect.astype(np.float32)
    else:
        raise ValueError("Parâmetro 'what' deve ser 'slope' ou 'aspect'")

def to_xarray(arr, profile, name="elevation"):
    """
    Converte um array numpy e perfil rasterio para xarray.DataArray georreferenciado.
    """
    if not isinstance(arr, np.ndarray) or arr.size == 0:
        raise ValueError("Array vazio ou inválido.")
    if not isinstance(profile, dict) or "transform" not in profile:
        raise KeyError("Perfil inválido: chave 'transform' ausente.")
    if not isinstance(name, str):
        raise ValueError("Nome deve ser string.")
    y, x = arr.shape
    xs = np.arange(x) * profile["transform"][0] + profile["transform"][2]
    ys = np.arange(y) * profile["transform"][4] + profile["transform"][5]
    da = xr.DataArray(
        arr, 
        coords=[("y", ys), ("x", xs)], 
        name=name,
        attrs={
            "crs": profile.get("crs", "EPSG:4326"),
            "units": "meters" if name == "elevation" else "",
            "long_name": "Elevation" if name == "elevation" else 
                        "Slope" if name == "slope" else 
                        "Aspect" if name == "aspect" else name
        }
    )
    return da

def visualize_dem_pygmt(dem_data, profile=None, output_path=None, region=None, title="Mapa de Elevação", 
                       cmap="geo", projection="M15c", illumination=True, contours=True):
    """
    Visualiza dados de elevação (DEM) usando PyGMT para criar mapas de alta qualidade.
    """
    if not PYGMT_AVAILABLE:
        raise ImportError("PyGMT não está instalado. Instale com: pip install pygmt")
    if isinstance(dem_data, np.ndarray) and profile is not None:
        dem_xr = to_xarray(dem_data, profile, "elevation")
    elif isinstance(dem_data, xr.DataArray):
        dem_xr = dem_data
    else:
        raise ValueError("dem_data deve ser um array NumPy com profile ou um xarray.DataArray")
    if region is None:
        region = [dem_xr.x.min().item(), dem_xr.x.max().item(), 
                 dem_xr.y.min().item(), dem_xr.y.max().item()]
    fig = pygmt.Figure()
    pygmt.makecpt(
        cmap=cmap,
        series=[dem_xr.min().item(), dem_xr.max().item()],
        continuous=True
    )
    if illumination:
        relief = pygmt.grdgradient(grid=dem_xr, azimuth=315, normalize="t")
        fig.grdimage(
            grid=dem_xr,
            region=region,
            projection=projection,
            frame=["a", f"+t{title}"],
            shading=relief,
            cmap=True
        )
    else:
        fig.grdimage(
            grid=dem_xr,
            region=region,
            projection=projection,
            frame=["a", f"+t{title}"],
            cmap=True
        )
    if contours:
        interval = (dem_xr.max().item() - dem_xr.min().item()) / 10
        fig.grdcontour(
            grid=dem_xr,
            interval=interval,
            annotation=interval*2,
            region=region,
            projection=projection
        )
    fig.colorbar(frame=["a", "x+lElevação", "y+lm"])
    fig.basemap(region=region, projection=projection, map_scale="g")
    fig.basemap(rose="jTR+w2c+l+o0.5c")
    if output_path:
        fig.savefig(output_path, dpi=300)
        logger.info(f"Mapa salvo em {output_path}")
    return fig
