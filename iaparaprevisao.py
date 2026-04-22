# --- Bibliotecas Necessárias ---
import os
import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.transform import Affine
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import contextily as ctx
from shapely.geometry import Point
import geopandas as gpd
import joblib
import warnings
import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict

# Import para pós-processamento. Instale com: pip install scikit-image
from skimage.morphology import remove_small_objects, remove_small_holes

# --- Configuração ---
# Mude para logging.DEBUG para ver a busca por cada arquivo
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# --- Constantes e Caminhos (AJUSTE ESTES CAMINHOS CONFORME NECESSÁRIO) ---
DEM_FILE_PATH = Path(r"D:\PESQUISA UFCA\DADOS TREINAMENTO\anadem_v1_24M.tif")
WATER_MASKS_FOLDER = Path(r"D:\PESQUISA UFCA\DADOS TREINAMENTO\GEE_Exports_Flood_Finder-Juazeiro_do_Norte_Eventos_Especificos")
PRECIPITATION_CSV_PATH = Path(r"D:\PESQUISA UFCA\DADOS TREINAMENTO\dados_tratados_ANA.csv")
LULC_FILE_PATH = Path(r"D:\PESQUISA UFCA\DADOS TREINAMENTO\GEE_Exports_Flood_Finder-Juazeiro_do_Norte-Baseline_DynamicWorld\baseline_dw_cover_juazeiro_do_norte_2022-08-01_a_2022-10-31.tif")

MODEL_SAVE_PATH = Path("modelo_previsao_inundacao_v4_otimizado.joblib")
RESULTS_FOLDER = Path("Resultados_Modelo_Inundacao_v4_Otimizado")
RESULTS_FOLDER.mkdir(exist_ok=True)

ORIGINAL_AOI_COORDS = [-39.3136, -7.2126]
ORIGINAL_AOI_BUFFER_METERS = 15000
TARGET_RESOLUTION_METERS = 10

# --- Funções Auxiliares ---

def create_aoi_from_point_and_buffer(lon: float, lat: float, buffer_meters: int, target_crs: str) -> Optional[gpd.GeoDataFrame]:
    """Cria uma Área de Interesse (AOI) a partir de um ponto central e um buffer."""
    try:
        point_geom = Point(lon, lat)
        point_gdf = gpd.GeoDataFrame([{'geometry': point_geom}], crs="EPSG:4326")
        estimated_utm_crs_for_buffer = point_gdf.estimate_utm_crs()
        if estimated_utm_crs_for_buffer is None:
            logger.warning("Não foi possível estimar o CRS UTM. Usando EPSG:3857.")
            crs_for_buffer = "EPSG:3857"
        else:
            crs_for_buffer = estimated_utm_crs_for_buffer
        point_gdf_metric = point_gdf.to_crs(crs_for_buffer)
        aoi_metric_geoseries = point_gdf_metric.buffer(buffer_meters, cap_style=3)
        aoi_gdf_target_crs = gpd.GeoDataFrame(geometry=aoi_metric_geoseries, crs=crs_for_buffer).to_crs(target_crs)
        return aoi_gdf_target_crs
    except Exception as e:
        logger.error(f"Erro ao criar AOI: {e}")
        return None

def load_and_align_raster(raster_path: Path, target_crs: rasterio.crs.CRS, target_transform: rasterio.Affine, target_width: int, target_height: int, resampling_method: Resampling, nodata_value_to_fill: Optional[float] = None) -> Optional[np.ndarray]:
    """Carrega um raster e o reprojeta para um grid de destino (template)."""
    try:
        with rasterio.open(raster_path) as src:
            if src.count == 0:
                logger.error(f"O raster {raster_path.name} está vazio.")
                return None
            src_nodata = src.nodata
            if nodata_value_to_fill is not None:
                dst_nodata = nodata_value_to_fill
            elif src_nodata is not None:
                dst_nodata = src_nodata
            else:
                dst_nodata = None
            destination_array = np.empty((target_height, target_width), dtype=src.dtypes[0])
            if dst_nodata is not None:
                destination_array.fill(dst_nodata)
            reproject(
                source=rasterio.band(src, 1),
                destination=destination_array,
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=src_nodata,
                dst_transform=target_transform,
                dst_crs=target_crs,
                dst_nodata=dst_nodata,
                resampling=resampling_method
            )
            logger.debug(f"Raster {raster_path.name} carregado e alinhado. Shape: {destination_array.shape}")
            return destination_array
    except FileNotFoundError:
        logger.error(f"Arquivo raster não encontrado: {raster_path}"); return None
    except Exception as e:
        logger.error(f"Erro ao carregar e alinhar raster {raster_path.name}: {e}"); return None

def calculate_slope(dem_array: np.ndarray, resolution_x: float, resolution_y: float) -> Optional[np.ndarray]:
    """Calcula a declividade (em graus) a partir de um array DEM."""
    if dem_array is None: return None
    if dem_array.ndim != 2 or dem_array.shape[0] < 2 or dem_array.shape[1] < 2:
        logger.error(f"Shape do DEM ({dem_array.shape}) inválido para cálculo de declividade.")
        return None
    try:
        dz_dy, dz_dx = np.gradient(dem_array, abs(resolution_y), abs(resolution_x))
        slope = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2)) * (180.0 / np.pi)
        logger.debug("Declividade calculada.")
        return slope
    except Exception as e:
        logger.error(f"Erro ao calcular declividade: {e}"); return None

def load_precipitation_data(csv_path: Path) -> Optional[pd.DataFrame]:
    """Carrega e limpa dados de precipitação de um arquivo CSV."""
    try:
        df_initial = pd.read_csv(csv_path, sep=None, engine='python', on_bad_lines='warn', dtype=str)
        if len(df_initial.columns) == 1 and ';' in df_initial.columns[0]:
            df_initial = pd.read_csv(csv_path, sep=';', on_bad_lines='warn', dtype=str)
        df = df_initial.copy()
        if 'Data' in df.columns and 'precipitacao_mm' in df.columns:
            df = df.rename(columns={'Data': 'data_col_orig', 'precipitacao_mm': 'precip_col_orig'})
        elif len(df.columns) >= 2:
            df = df.rename(columns={df.columns[0]: 'data_col_orig', df.columns[1]: 'precip_col_orig'})
        else:
            logger.error(f"Não foi possível identificar colunas de data/precipitação em {csv_path}."); return None
        df['data_col_parsed'] = pd.to_datetime(df['data_col_orig'], errors='coerce', dayfirst=True)
        df.dropna(subset=['data_col_parsed'], inplace=True)
        df['date_str'] = df['data_col_parsed'].dt.strftime('%Y-%m-%d')
        df['precipitacao_mm'] = pd.to_numeric(df['precip_col_orig'].str.replace(',', '.'), errors='coerce')
        df.dropna(subset=['precipitacao_mm'], inplace=True)
        logger.info(f"Dados de precipitação carregados de {csv_path}. Linhas válidas: {len(df)}")
        return df[['date_str', 'precipitacao_mm']]
    except Exception as e:
        logger.error(f"Erro crítico ao carregar dados de precipitação de {csv_path}: {e}", exc_info=True); return None

# --- Funções de Machine Learning ---

def prepare_training_data(dem_path: Path, lulc_path: Path, water_masks_dir: Path, precipitation_df: pd.DataFrame, aoi_coords_lon_lat: list, aoi_buffer_m: int, target_resolution_m: float) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[rasterio.Affine], Optional[rasterio.crs.CRS], Optional[tuple], Optional[list]]:
    """Prepara os dados de treinamento para o modelo de inundação."""
    logger.info("--- Iniciando Preparação dos Dados de Treinamento ---")

    # Passo 1: Definir Grid de Referência (Template)
    try:
        with rasterio.open(dem_path) as dem_src:
            dem_nodata_val = dem_src.nodata or -9999.0
        point_gdf = gpd.GeoDataFrame([{'geometry': Point(aoi_coords_lon_lat[0], aoi_coords_lon_lat[1])}], crs="EPSG:4326")
        template_crs_utm = point_gdf.estimate_utm_crs()
        if not template_crs_utm:
            logger.warning("Estimativa UTM CRS falhou. Usando EPSG:3857."); template_crs_utm = "EPSG:3857"
        
        aoi_gdf_utm = create_aoi_from_point_and_buffer(aoi_coords_lon_lat[0], aoi_coords_lon_lat[1], aoi_buffer_m, template_crs_utm.to_string())
        if aoi_gdf_utm is None: return None, None, None, None, None, None
        
        dst_bounds = aoi_gdf_utm.total_bounds
        template_width = int(np.ceil((dst_bounds[2] - dst_bounds[0]) / target_resolution_m))
        template_height = int(np.ceil((dst_bounds[3] - dst_bounds[1]) / target_resolution_m))
        template_transform = Affine(target_resolution_m, 0.0, dst_bounds[0], 0.0, -target_resolution_m, dst_bounds[3])
        template_shape = (template_height, template_width)
        logger.info(f"Grid de referência definido: CRS={template_crs_utm.to_string()}, Shape={template_shape}")
    except Exception as e:
        logger.error(f"Erro ao criar template: {e}"); return None, None, None, None, None, None

    # Passo 2: Carregar e Alinhar Rasters Base
    dem_aoi_data = load_and_align_raster(dem_path, template_crs_utm, template_transform, template_width, template_height, Resampling.bilinear, dem_nodata_val)
    lulc_aoi_data = load_and_align_raster(lulc_path, template_crs_utm, template_transform, template_width, template_height, Resampling.nearest, 0)
    if dem_aoi_data is None or lulc_aoi_data is None: return None, None, None, None, None, None

    # Passo 3: Engenharia de Features
    slope_aoi_data = calculate_slope(dem_aoi_data, abs(template_transform.a), abs(template_transform.e))
    valid_pixel_mask = (dem_aoi_data != dem_nodata_val) & np.isfinite(dem_aoi_data)

    base_features_list = [dem_aoi_data[valid_pixel_mask], lulc_aoi_data[valid_pixel_mask]]
    feature_names = ['altitude', 'uso_solo']

    if slope_aoi_data is not None:
        base_features_list.insert(1, slope_aoi_data[valid_pixel_mask])
        feature_names.insert(1, 'declividade')

    # Passo 4: Processar Eventos de Inundação (COM A CORREÇÃO DO NOME DO ARQUIVO)
    all_X_events, all_y_events = [], []
    for _, row in precipitation_df.iterrows():
        date_str, precip_value = row['date_str'], row['precipitacao_mm']
        
        potential_path = water_masks_dir / f"mascara_agua_Juazeiro_do_Norte_S1_{date_str}.tif"
        logger.debug(f"Procurando por: {potential_path}")

        found_mask_file = None
        if potential_path.exists():
            found_mask_file = potential_path
        
        if found_mask_file:
            logger.info(f"Processando evento: {date_str}, Precip: {precip_value}mm, Máscara: {found_mask_file.name}")
            water_mask_data = load_and_align_raster(found_mask_file, template_crs_utm, template_transform, template_width, template_height, Resampling.nearest, 0)
            if water_mask_data is None: continue
            
            labels_flat_valid = (water_mask_data > 0).astype(np.uint8)[valid_pixel_mask]
            precip_feature_array = np.full(dem_aoi_data[valid_pixel_mask].shape[0], precip_value, dtype=np.float32)
            
            current_event_features = np.column_stack(base_features_list + [precip_feature_array])
            
            all_X_events.append(current_event_features)
            all_y_events.append(labels_flat_valid)

    if not all_X_events:
        logger.error("Nenhum dado de evento de treinamento foi preparado. Verifique o padrão de nome dos arquivos de máscara ('S1' etc.) e as datas no CSV."); return None, None, None, None, None, None

    X_final = np.concatenate(all_X_events, axis=0)
    y_final = np.concatenate(all_y_events, axis=0)
    final_feature_names = feature_names + ['precipitacao']
    
    logger.info(f"Preparação de dados concluída. Total de eventos: {len(all_X_events)}. Shape final: X{X_final.shape}, y{y_final.shape}")
    return X_final, y_final, template_transform, template_crs_utm, template_shape, final_feature_names

def train_and_evaluate_model(X: np.ndarray, y: np.ndarray, feature_names: list, save_path: Optional[Path] = None) -> Optional[RandomForestClassifier]:
    """
    Treina um modelo RandomForestClassifier usando GridSearchCV para encontrar os melhores
    hiperparâmetros e avalia seu desempenho.
    """
    if X is None or y is None or X.shape[0] == 0:
        logger.error("Dados de treinamento inválidos fornecidos."); return None
    
    if len(np.unique(y)) < 2:
        logger.warning("Rótulos (y) têm menos de duas classes. O modelo pode não aprender bem."); return None

    logger.info(f"Dividindo dados em treino (70%) e teste (30%)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    logger.info("Configurando a busca de hiperparâmetros com GridSearchCV...")
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [25, 35, None],
        'min_samples_split': [10, 20],
        'min_samples_leaf': [5, 10],
        'max_features': ['sqrt', 'log2']
    }

    rf = RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2, scoring='f1_weighted')

    logger.info(f"Iniciando o treinamento com GridSearchCV para as features: {feature_names}...")
    try:
        grid_search.fit(X_train, y_train)
        
        logger.info(f"Melhores parâmetros encontrados: {grid_search.best_params_}")
        best_model = grid_search.best_estimator_

        logger.info("Avaliando o melhor modelo nos dados de teste...")
        y_pred_test = best_model.predict(X_test)
        
        report = classification_report(y_test, y_pred_test, target_names=['Não Inundado', 'Inundado'], zero_division=0)
        logger.info("Relatório de Classificação (Teste):\n" + report)

        cm = confusion_matrix(y_test, y_pred_test)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Não Inundado', 'Inundado'])
        disp.plot(cmap=plt.cm.Blues); plt.title("Matriz de Confusão (Dados de Teste)")
        
        if save_path:
            report_path = save_path.parent / (save_path.stem + "_classification_report.txt")
            with open(report_path, "w") as f: f.write(f"Melhores Parâmetros: {grid_search.best_params_}\n\n{report}")
            logger.info(f"Relatório de classificação salvo em: {report_path}")
            
            cm_path = save_path.parent / (save_path.stem + "_confusion_matrix.png")
            plt.savefig(cm_path); plt.close()
            logger.info(f"Matriz de confusão salva em: {cm_path}")
            
            joblib.dump(best_model, save_path)
            logger.info(f"Melhor modelo salvo em: {save_path}")

        return best_model
    except Exception as e:
        logger.error(f"Erro durante o GridSearchCV: {e}", exc_info=True); return None

def predict_flood_extent(model: RandomForestClassifier, dem_data: np.ndarray, slope_data: Optional[np.ndarray], lulc_data: np.ndarray, precip_mm: float, valid_mask: np.ndarray, model_features: list, prob_threshold: Optional[float] = None) -> Optional[np.ndarray]:
    """Prevê a extensão da inundação para um novo evento."""
    logger.info(f"Preparando features para previsão (Precip: {precip_mm}mm)...")
    
    features_map = {
        'altitude': dem_data[valid_mask],
        'uso_solo': lulc_data[valid_mask],
        'precipitacao': np.full(dem_data[valid_mask].shape[0], precip_mm, dtype=np.float32)
    }
    if slope_data is not None and 'declividade' in model_features:
        features_map['declividade'] = slope_data[valid_mask]
    
    # Garante a ordem correta das features
    X_predict_list = []
    for feat_name in model_features:
        if feat_name in features_map:
            X_predict_list.append(features_map[feat_name])
        else:
            logger.error(f"Feature esperada pelo modelo '{feat_name}' não encontrada para predição.")
            return None

    X_predict = np.column_stack(X_predict_list)
    if X_predict.shape[0] == 0:
        logger.warning("Nenhum pixel válido para previsão."); return np.zeros(dem_data.shape, dtype=np.uint8)
    
    logger.info(f"Realizando predição para {X_predict.shape[0]} pixels...")
    try:
        if prob_threshold is not None:
            logger.info(f"Usando limiar de probabilidade customizado: {prob_threshold}")
            y_pred_probs = model.predict_proba(X_predict)[:, 1]
            y_pred_pixels = (y_pred_probs >= prob_threshold).astype(np.uint8)
        else:
            logger.info("Usando model.predict() com limiar padrão.")
            y_pred_pixels = model.predict(X_predict)
            
        prediction_mask_2d = np.zeros(dem_data.shape, dtype=np.uint8)
        prediction_mask_2d[valid_mask] = y_pred_pixels
        logger.info("Previsão da mancha de inundação concluída.")
        return prediction_mask_2d
    except Exception as e:
        logger.error(f"Erro durante a predição: {e}", exc_info=True); return None

def post_process_prediction(prediction_mask: np.ndarray, min_object_size: int = 50, min_hole_size: int = 100) -> np.ndarray:
    """Limpa a máscara de previsão removendo pequenos objetos e preenchendo buracos."""
    logger.info(f"Iniciando pós-processamento: removendo objetos < {min_object_size}px e buracos < {min_hole_size}px.")
    if not np.any(prediction_mask):
        logger.warning("Máscara de previsão vazia. Pós-processamento pulado.")
        return prediction_mask

    bool_mask = prediction_mask.astype(bool)
    cleaned_mask = remove_small_objects(bool_mask, min_size=min_object_size)
    final_mask = remove_small_holes(cleaned_mask, area_threshold=min_hole_size)
    
    logger.info("Pós-processamento concluído.")
    return final_mask.astype(np.uint8)

def plot_actual_vs_predicted_map(flood_data_affine: rasterio.Affine, flood_data_shape: tuple, actual_flood_mask: Optional[np.ndarray], predicted_flood_mask: np.ndarray, flood_data_crs, title: str, save_path: Path):
    """Plota um mapa comparando a inundação real com a prevista, lado a lado."""
    fig, axes = plt.subplots(1, 2, figsize=(22, 10), sharex=True, sharey=True)
    ax1, ax2 = axes.ravel()
    
    plot_extent = [flood_data_affine.c, flood_data_affine.c + flood_data_affine.a * flood_data_shape[1],
                   flood_data_affine.f + flood_data_affine.e * flood_data_shape[0], flood_data_affine.f]

    ax1.set_title("Previsão de Inundação (Pós-Processada)", fontsize=14)
    predicted_masked = np.ma.masked_where(predicted_flood_mask == 0, predicted_flood_mask)
    ax1.imshow(predicted_masked, cmap='Reds', alpha=0.7, extent=plot_extent, zorder=10)
    ctx.add_basemap(ax1, crs=flood_data_crs.to_string(), source=ctx.providers.OpenStreetMap.Mapnik)
    
    ax2.set_title("Inundação Real (Observada)", fontsize=14)
    if actual_flood_mask is not None and np.any(actual_flood_mask):
        actual_masked = np.ma.masked_where(actual_flood_mask == 0, actual_flood_mask)
        ax2.imshow(actual_masked, cmap='Blues', alpha=0.7, extent=plot_extent, zorder=10)
    else:
        ax2.text(0.5, 0.5, "Máscara real não disponível", ha='center', va='center', transform=ax2.transAxes)
    ctx.add_basemap(ax2, crs=flood_data_crs.to_string(), source=ctx.providers.OpenStreetMap.Mapnik)

    fig.suptitle(title, fontsize=18, y=0.96)
    legend_patches = [mpatches.Patch(color='red', alpha=0.7, label='Inundação Prevista'),
                      mpatches.Patch(color='blue', alpha=0.7, label='Inundação Real')]
    fig.legend(handles=legend_patches, loc='lower center', ncol=2, bbox_to_anchor=(0.5, 0.01))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.93])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Mapa comparativo salvo em: {save_path}")
    plt.close(fig)

# --- Função Principal ---
def main():
    logger.info("--- Iniciando Script de Previsão de Inundação v4 (Otimizado) ---")
    
    precipitation_data = load_precipitation_data(PRECIPITATION_CSV_PATH)
    if precipitation_data is None: return

    FULL_MODEL_SAVE_PATH = RESULTS_FOLDER / MODEL_SAVE_PATH

    X_data, y_data, template_affine, template_crs, template_shape, feature_names = prepare_training_data(
        dem_path=DEM_FILE_PATH, lulc_path=LULC_FILE_PATH, water_masks_dir=WATER_MASKS_FOLDER,
        precipitation_df=precipitation_data, aoi_coords_lon_lat=ORIGINAL_AOI_COORDS,
        aoi_buffer_m=ORIGINAL_AOI_BUFFER_METERS, target_resolution_m=TARGET_RESOLUTION_METERS
    )
    if template_affine is None:
        logger.error("Falha na preparação de dados/template. Encerrando."); return

    ### INÍCIO DA CORREÇÃO ###
    # Bloco para reduzir o tamanho do dataset e evitar erros de memória.
    # Ele pega uma amostra aleatória dos dados, mas mantém a mesma proporção
    # de pixels inundados e não-inundados do dataset original.
    # Ajuste SAMPLE_SIZE conforme a memória RAM disponível no seu computador.
    SAMPLE_SIZE = 1_500_000 # Usando 1.5 milhões de pixels como exemplo.

    if X_data is not None and X_data.shape[0] > SAMPLE_SIZE:
        logger.info(f"Dataset original com {X_data.shape[0]} amostras. Reduzindo para {SAMPLE_SIZE} via amostragem estratificada.")
        # Usamos train_test_split como um "truque" para fazer a amostragem estratificada de forma fácil.
        # Descartamos o restante dos dados com `_`.
        X_data, _, y_data, _ = train_test_split(
            X_data, y_data,
            train_size=SAMPLE_SIZE,
            random_state=42,
            stratify=y_data  # Parâmetro CRÍTICO para manter a proporção das classes.
        )
        logger.info(f"Novo shape dos dados para treinamento: X{X_data.shape}, y{y_data.shape}")
    ### FIM DA CORREÇÃO ###


    flood_model, model_features = None, feature_names
    if FULL_MODEL_SAVE_PATH.exists():
        logger.info(f"Carregando modelo pré-treinado de {FULL_MODEL_SAVE_PATH}...")
        try:
            flood_model = joblib.load(FULL_MODEL_SAVE_PATH)
            model_features = flood_model.feature_names_in_.tolist()
            logger.info(f"Modelo carregado. Features do modelo: {model_features}")
        except Exception as e:
            logger.error(f"Erro ao carregar modelo: {e}. Um novo modelo será treinado.")
    
    if flood_model is None:
        if X_data is None or y_data is None or feature_names is None:
            logger.error("Dados de treinamento não disponíveis. Não é possível treinar novo modelo."); return
        
        logger.info("--- Iniciando Fase de Treinamento e Avaliação do Modelo com GridSearchCV ---")
        flood_model = train_and_evaluate_model(X_data, y_data, feature_names, save_path=FULL_MODEL_SAVE_PATH)
        if flood_model is None:
            logger.error("Falha no treinamento do modelo. Encerrando."); return
        # A linha "model_features = flood_model.feature_names_in_.tolist()" foi removida.

    logger.info("--- Iniciando Fase de Exemplo de Previsão e Visualização ---")
    
    # Pega o último evento do CSV como exemplo para visualização
    example_event = precipitation_data.iloc[-1]
    example_precip_val = example_event['precipitacao_mm']
    example_date_str = example_event['date_str']
    example_actual_mask_path = WATER_MASKS_FOLDER / f"mascara_agua_Juazeiro_do_Norte_S1_{example_date_str}.tif"
    
    logger.info(f"Evento de exemplo para visualização: Data={example_date_str}, Precipitação={example_precip_val:.1f}mm")

    dem_pred, lulc_pred = (
        load_and_align_raster(p, template_crs, template_affine, template_shape[1], template_shape[0], m, v)
        for p, m, v in [
            (DEM_FILE_PATH, Resampling.bilinear, (rasterio.open(DEM_FILE_PATH).nodata or -9999.0)),
            (LULC_FILE_PATH, Resampling.nearest, 0)
        ]
    )
    if dem_pred is None or lulc_pred is None:
        logger.error("Falha ao recarregar dados para a predição. Encerrando."); return

    dem_nodata = rasterio.open(DEM_FILE_PATH).nodata or -9999.0
    valid_pixels_pred = (dem_pred != dem_nodata) & np.isfinite(dem_pred)
    slope_pred = calculate_slope(dem_pred, abs(template_affine.a), abs(template_affine.e))
    
    predicted_mask_raw = predict_flood_extent(
        model=flood_model, dem_data=dem_pred, slope_data=slope_pred, lulc_data=lulc_pred,
        precip_mm=example_precip_val, valid_mask=valid_pixels_pred, model_features=model_features,
        prob_threshold=0.6 # Limiar de probabilidade para classificar um pixel como inundado
    )

    if predicted_mask_raw is not None:
        predicted_mask_processed = post_process_prediction(predicted_mask_raw, min_object_size=50, min_hole_size=100)
        
        actual_mask_for_plot = None
        if example_actual_mask_path.exists():
            actual_data = load_and_align_raster(example_actual_mask_path, template_crs, template_affine, template_shape[1], template_shape[0], Resampling.nearest, 0)
            if actual_data is not None:
                actual_mask_for_plot = (actual_data > 0).astype(np.uint8)

        plot_title = f"Comparação Inundação: Real vs. Prevista\nData: {example_date_str}, Precip: {example_precip_val:.1f}mm"
        plot_save_path = RESULTS_FOLDER / f"mapa_comparacao_{example_date_str}.png"
        
        plot_actual_vs_predicted_map(
            template_affine, template_shape, actual_mask_for_plot,
            predicted_mask_processed,
            template_crs, plot_title, plot_save_path
        )
    else:
        logger.error("Falha ao gerar a máscara de inundação prevista.")

    logger.info("--- Script de Previsão de Inundação Concluído ---")

if __name__ == "__main__":
    main()