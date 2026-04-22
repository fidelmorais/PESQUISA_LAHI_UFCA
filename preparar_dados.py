# --- Bibliotecas Necessárias ---
# pip install earthengine-api pandas pathlib openpyxl
import ee
import pandas as pd
from datetime import datetime, timedelta
import logging
import json
import time
from typing import Dict, List, Optional
from pathlib import Path

# --- Configuração do Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==============================================================================
# --- ÁREA DE CONFIGURAÇÃO (AJUSTE ESTES VALORES) ---
# ==============================================================================

# Caminho para o seu arquivo com as datas e precipitação
PRECIPITATION_DATA_PATH = Path(r"D:\PESQUISA UFCA\DADOS TREINAMENTO\dados_tratados_ANA.xlsx") # MANTENHA O SEU CAMINHO

# Colunas esperadas no arquivo de dados.
DATE_COLUMN_NAME = 'Data'
PRECIP_COLUMN_NAME = 'precipitacao_mm'

# Limiar de precipitação (em mm) para processar uma data.
PRECIPITATION_THRESHOLD = 0.0

# Janela de análise em dias após o evento de chuva.
ANALYSIS_WINDOW_DAYS = 5

# Para Sentinel-1 (Radar)
RADAR_OPTIMAL_THRESHOLD_DB = -19.0 # Limiar que você identificou como ótimo
FOCAL_MEDIAN_RADIUS_S1 = 50      # Raio do filtro de speckle em metros

# --- Configurações da Área e Exportação ---
LOCATION_NAME = "Juazeiro_do_Norte"
COORDS = [-39.3136, -7.2126] # Formato: [Lon, Lat]
BUFFER_METERS = 15000 
GEE_PROJECT_ID = 'pesquisa-lhiufca' 
GEE_EXPORT_FOLDER = f"GEE_Exports_S1_Flood_Finder/{LOCATION_NAME}_Eventos_Especificos"
OUTPUT_DIR = Path(f"Resultados_Por_Evento_S1")

# ==============================================================================
# --- INICIALIZAÇÃO E FUNÇÕES AUXILIARES ---
# ==============================================================================

try:
    ee.Initialize(project=GEE_PROJECT_ID, opt_url='https://earthengine-highvolume.googleapis.com')
    logger.info(f"Google Earth Engine inicializado com sucesso no projeto '{GEE_PROJECT_ID}' (endpoint de alto volume).")
except Exception:
    try:
        ee.Authenticate()
        ee.Initialize(project=GEE_PROJECT_ID)
        logger.info(f"Google Earth Engine inicializado com sucesso no projeto '{GEE_PROJECT_ID}'.")
    except Exception as e_init:
        logger.critical(f"Erro crítico ao inicializar o Earth Engine: {e_init}")
        raise

def load_event_data(file_path: Path, date_col: str, precip_col: str, threshold: float) -> Optional[pd.DataFrame]:
    """Lê dados de um arquivo CSV ou Excel, valida colunas e filtra por precipitação."""
    try:
        if not file_path.exists():
            logger.error(f"Arquivo de dados não encontrado em: {file_path}"); return None

        if file_path.suffix.lower() == '.csv':
            df = pd.read_csv(file_path, header=0, encoding='latin-1', delimiter=';')
        elif file_path.suffix.lower() in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path, header=0, engine='openpyxl')
        else:
            logger.error(f"Formato de arquivo não suportado: {file_path.suffix}"); return None

        df = df.rename(columns={date_col: 'event_date_str', precip_col: 'precipitacao_mm'})

        if 'event_date_str' not in df.columns or 'precipitacao_mm' not in df.columns:
            logger.error(f"Colunas '{date_col}' ou '{precip_col}' não encontradas no arquivo."); return None

        df['event_date'] = pd.to_datetime(df['event_date_str'], format='%d/%m/%Y', errors='coerce')
        df['precipitacao_mm'] = pd.to_numeric(df['precipitacao_mm'], errors='coerce')
        df.dropna(subset=['event_date', 'precipitacao_mm'], inplace=True)

        filtered_df = df[df['precipitacao_mm'] > threshold].copy()
        logger.info(f"Encontradas {len(filtered_df)} datas no arquivo com precipitação > {threshold}mm.")
        return filtered_df[['event_date', 'precipitacao_mm']]

    except Exception as e:
        logger.error(f"Erro ao ler o arquivo de dados '{file_path}': {e}"); return None

# ==============================================================================
# --- CLASSE DE ANÁLISE PRINCIPAL ---
# ==============================================================================

class EventWaterAnalyzer:
    """Analisador de corpos d'água usando imagens Sentinel-1 (Radar)."""
    
    RADAR_WATER_THRESHOLD_DB = RADAR_OPTIMAL_THRESHOLD_DB 
    FOCAL_MEDIAN_RADIUS_S1 = FOCAL_MEDIAN_RADIUS_S1
    SENTINEL1_SCALE = 10 # Resolução aproximada do Sentinel-1 GRD em metros

    def __init__(self, coordinates: List[float], buffer_size_meters: int):
        self.aoi = ee.Geometry.Point(coordinates).buffer(buffer_size_meters)
        logger.info(f"Analisador de eventos configurado para AOI com buffer de {buffer_size_meters}m.")

    def get_s1_water_mask(self, start_date: datetime, end_date: datetime) -> Optional[Dict]:
        """Cria uma máscara de água usando dados de Radar Sentinel-1."""
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        logger.info(f"[RADAR] Buscando imagens Sentinel-1 entre {start_date_str} e {end_date_str}")
        try:
            s1_collection = (ee.ImageCollection('COPERNICUS/S1_GRD')
                               .filterBounds(self.aoi)
                               .filterDate(start_date_str, end_date_str)
                               .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
                               .filter(ee.Filter.eq('instrumentMode', 'IW')))

            if s1_collection.size().getInfo() == 0:
                logger.warning(f"[RADAR] Nenhuma imagem Sentinel-1 encontrada na janela."); return None

            s1_image_median = s1_collection.median() 
            vh_band_processed = s1_image_median.select('VH').focal_median(
                self.FOCAL_MEDIAN_RADIUS_S1, 'circle', 'meters'
            )
            
            water_mask = vh_band_processed.lt(self.RADAR_WATER_THRESHOLD_DB).rename('water_mask').toByte()
            
            image_date_representative = start_date 

            logger.info(f"[RADAR] Máscara de água criada para a janela iniciando em {image_date_representative.strftime('%Y-%m-%d')}.")
            return {
                'image_mask': water_mask,
                'image_date': image_date_representative.strftime('%Y-%m-%d'), 
                'cloud_cover_percent': 'N/A (Radar)' # Radar não tem nuvens
            }
        except Exception as e:
            logger.error(f"[RADAR] Erro ao processar GEE com Sentinel-1: {e}"); return None

    def calculate_water_area_stats(self, water_mask_image: ee.Image) -> Dict:
        """Calcula a área (em m² e ha) coberta pela máscara de água."""
        try:
            area_image = water_mask_image.multiply(ee.Image.pixelArea())
            stats = area_image.reduceRegion(
                reducer=ee.Reducer.sum(),
                geometry=self.aoi,
                scale=self.SENTINEL1_SCALE, 
                maxPixels=1e10,
                bestEffort=True
            )
            water_area_m2 = ee.Number(stats.get('water_mask')).getInfo() or 0.0
            return {"water_area_m2": round(water_area_m2, 2), "water_area_ha": round(water_area_m2 / 10000, 2)}
        except Exception as e:
            logger.error(f"Erro ao calcular estatísticas de área: {e}"); return {"water_area_m2": 0, "water_area_ha": 0}

    def export_image_to_drive(self, image_to_export: ee.Image, file_name: str, folder_name: str) -> None:
        """Inicia uma tarefa de exportação da imagem para o Google Drive."""
        description = Path(file_name).stem
        task = ee.batch.Export.image.toDrive(
            image=image_to_export,
            description=description,
            folder=folder_name,
            fileNamePrefix=file_name,
            scale=self.SENTINEL1_SCALE, 
            region=self.aoi,
            fileFormat='GeoTIFF',
            maxPixels=1e10
        )
        task.start()
        logger.info(f"Tarefa de exportação para '{file_name}' iniciada. ID: {task.id}")

# ==============================================================================
# --- FUNÇÃO PRINCIPAL ---
# ==============================================================================
def main() -> None:
    """Função principal que orquestra todo o processo de análise."""
    logger.info("--- Iniciando Análise de Inundação com Sentinel-1 (Radar) ---")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True) 

    event_dates_df = load_event_data(PRECIPITATION_DATA_PATH, DATE_COLUMN_NAME, PRECIP_COLUMN_NAME, PRECIPITATION_THRESHOLD)
    if event_dates_df is None or event_dates_df.empty:
        logger.critical("Nenhuma data de evento para processar. Verifique o arquivo de dados. Encerrando."); return

    analyzer = EventWaterAnalyzer(coordinates=COORDS, buffer_size_meters=BUFFER_METERS)
    all_events_summary = []

    for _, row in event_dates_df.iterrows():
        event_date, precip = row['event_date'], row['precipitacao_mm']
        start_window, end_window = event_date, event_date + timedelta(days=ANALYSIS_WINDOW_DAYS)
        logger.info(f"--- Processando evento: Data={event_date.strftime('%Y-%m-%d')}, Precip.={precip:.1f}mm ---")
        
        logger.info("Usando método de análise via Radar (Sentinel-1).")
        result = analyzer.get_s1_water_mask(start_window, end_window)
        
        if result and result.get('image_mask'):
            water_mask = result['image_mask']
            stats = analyzer.calculate_water_area_stats(water_mask) # Scale é fixo na classe agora
            logger.info(f"Área de água detectada: {stats.get('water_area_ha', 0):.2f} ha")
            
            event_summary = {
                "event_date": event_date.strftime('%Y-%m-%d'),
                "precipitation_mm": precip,
                "analysis_satellite": "Sentinel-1 SAR",
                "analysis_window_start": start_window.strftime('%Y-%m-%d'),
                "analysis_window_end": end_window.strftime('%Y-%m-%d'),
                "selected_image_date": result['image_date'], # Data da mediana, representativa da janela
                "selected_image_cloud_cover": result['cloud_cover_percent'], # Sempre 'N/A (Radar)'
                "water_area_m2": stats['water_area_m2'],
                "water_area_ha": stats['water_area_ha']
            }
            all_events_summary.append(event_summary)
            
            output_filename_prefix = f"mascara_agua_{LOCATION_NAME}_S1_{event_date.strftime('%Y-%m-%d')}"
            analyzer.export_image_to_drive(water_mask, output_filename_prefix, GEE_EXPORT_FOLDER) # Scale é fixo na classe
            time.sleep(2) 
        else:
            logger.warning(f"Nenhuma imagem válida encontrada para o evento de {event_date.strftime('%Y-%m-%d')}. Pulando.")

    if all_events_summary:
        summary_path = OUTPUT_DIR / f"resumo_estatisticas_{LOCATION_NAME}_S1.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(all_events_summary, f, ensure_ascii=False, indent=4)
        logger.info(f"Resumo dos eventos salvo em: {summary_path}")

    logger.info("--- Processamento com Sentinel-1 concluído. Verifique o status das tarefas na interface do Google Earth Engine. ---")

if __name__ == "__main__":
    main()