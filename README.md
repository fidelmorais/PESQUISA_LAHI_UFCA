# PESQUISA_LAHI_UFCA — Previsão de Inundações com Machine Learning

Projeto de pesquisa do **LAHI (Laboratório de Recursos Hídricos e Inundações)** da **UFCA (Universidade Federal do Cariri)**, focado na previsão de manchas de inundação em **Juazeiro do Norte – CE** usando sensoriamento remoto e aprendizado de máquina.

---

## Estrutura do Projeto

```
PESQUISA_LAHI_UFCA/
├── preparar_dados.py                   # Etapa 1: extração de máscaras de água via GEE
├── iaparaprevisao.py                   # Etapa 2: treinamento e previsão com Random Forest
└── IMPLEMENTADOS_MAS_NAO_USADOS/      # Módulos experimentais / não integrados ao pipeline
    ├── ana_pluviometry.py              # Cliente para a API da ANA (Hidroweb)
    ├── logging_custom.py               # Utilitário de logging em arquivo
    └── topography.py                   # Download e processamento de DEM (Copernicus/GEE)
```

---

## Pipeline Principal

```
preparar_dados.py  →  máscaras GeoTIFF + resumo JSON
        ↓
iaparaprevisao.py  →  modelo treinado + mapas comparativos
```

---

## Arquivos Ativos

### `preparar_dados.py` — Preparação dos Dados de Inundação

Conecta-se ao **Google Earth Engine** e, para cada evento de precipitação registrado em um arquivo CSV/XLSX, gera uma máscara binária de água a partir de imagens **Sentinel-1 SAR (banda VH)**.

**O que faz:**
- Lê datas e precipitação do arquivo de eventos (`dados_tratados_ANA.xlsx`).
- Para cada evento, filtra imagens Sentinel-1 numa janela de 5 dias após a chuva.
- Aplica filtro de speckle (mediana focal, raio 50 m) e classifica pixels como água usando limiar de −19 dB.
- Calcula a área inundada (m² e ha) via `ee.Reducer`.
- Exporta cada máscara como GeoTIFF para o Google Drive.
- Salva um `resumo_estatisticas_*.json` com os metadados de todos os eventos.

**Saída:** arquivos `mascara_agua_Juazeiro_do_Norte_S1_YYYY-MM-DD.tif` + JSON de resumo.

**Configurações principais (ajustáveis no topo do arquivo):**

| Parâmetro | Valor padrão |
|---|---|
| Localização | Juazeiro do Norte (−39.3136, −7.2126) |
| Buffer da AOI | 15 km |
| Limiar SAR | −19.0 dB |
| Janela de análise | 5 dias após o evento |
| Resolução de exportação | 10 m |

---

### `iaparaprevisao.py` — Treinamento e Previsão com Random Forest

Usa as máscaras geradas pelo script anterior para treinar um classificador **Random Forest** que prevê pixels inundados a partir de atributos do terreno e precipitação.

**O que faz:**
- Define um grid de referência de 10 m sobre a AOI projetado em UTM.
- Carrega e alinha ao grid: DEM (ANADEM v1), LULC (Dynamic World) e máscaras de água.
- Calcula declividade a partir do DEM com `np.gradient`.
- Monta o dataset de treino: features `[altitude, declividade, uso_solo, precipitacao]` + rótulo binário (inundado/não inundado).
- Reduz o dataset a 1,5 M amostras estratificadas para caber em memória.
- Otimiza hiperparâmetros com **GridSearchCV** (validação cruzada 3-fold, métrica F1 ponderada).
- Avalia o melhor modelo (relatório de classificação + matriz de confusão).
- Prevê a mancha de inundação para um evento de exemplo.
- Pós-processa a previsão (remove objetos < 50 px e preenche buracos < 100 px via `skimage`).
- Plota mapa comparativo Previsto × Real com basemap OpenStreetMap.

**Saída:** modelo `.joblib`, relatório `.txt`, matriz de confusão `.png`, mapa comparativo `.png` — tudo em `Resultados_Modelo_Inundacao_v4_Otimizado/`.

**Caminhos que precisam ser ajustados antes de rodar:**

```python
DEM_FILE_PATH          = Path(r"<caminho>/anadem_v1_24M.tif")
WATER_MASKS_FOLDER     = Path(r"<caminho>/GEE_Exports_...")
PRECIPITATION_CSV_PATH = Path(r"<caminho>/dados_tratados_ANA.csv")
LULC_FILE_PATH         = Path(r"<caminho>/baseline_dw_cover_*.tif")
```

---

## Módulos Experimentais (`IMPLEMENTADOS_MAS_NAO_USADOS/`)

Estes módulos foram desenvolvidos durante a pesquisa mas **não estão integrados ao pipeline principal**.

### `ana_pluviometry.py` — Cliente da API ANA/Hidroweb

Funções para autenticação e consulta à API REST do Hidroweb da **Agência Nacional de Águas (ANA)**:

- `obter_token_ana()` — obtém token JWT via endpoint de autenticação.
- `buscar_inventario_estacoes()` — lista todas as estações telemetradas.
- `filtrar_estacoes_geograficamente()` — filtra estações por bounding box e/ou raio em km.
- `buscar_serie_estacao()` — busca série histórica de precipitação por estação e intervalo de datas.
- `obter_chuva_diaria_em_intervalo()` — agrega chuva diária de múltiplas estações em um DataFrame.
- `make_dataset()` — combina séries de todas as estações num `xr.Dataset`.

> **Nota:** as credenciais `ANA_USER` / `ANA_PASS` no topo do arquivo devem ser substituídas pelas suas credenciais pessoais antes de usar.

### `logging_custom.py` — Utilitário de Logging

Expõe uma única função `create_logger()` que cria um logger configurado para gravar em arquivo e, opcionalmente, no console — com formatação padronizada para o projeto.

### `topography.py` — Download e Processamento de DEM

Funções para obter e processar Modelos Digitais de Elevação:

- `fetch_dem()` — baixa e mosaica tiles **Copernicus DEM 30 m** (via HTTP/S3) ou exporta via **Google Earth Engine**; suporta cache local.
- `slope_aspect()` — calcula declividade ou aspecto a partir de um array DEM.
- `to_xarray()` — converte array NumPy + perfil rasterio em `xr.DataArray` georreferenciado.
- `visualize_dem_pygmt()` — gera mapa topográfico de alta qualidade com iluminação, curvas de nível e barra de cores usando **PyGMT**.

---

## Dependências Principais

```
earthengine-api   # Google Earth Engine
rasterio          # leitura/escrita de rasters
geopandas         # geometrias vetoriais
scikit-learn      # Random Forest, GridSearchCV
scikit-image      # pós-processamento morfológico
pandas / numpy    # manipulação de dados
matplotlib        # visualização
contextily        # basemap OpenStreetMap
joblib            # serialização do modelo
openpyxl          # leitura de arquivos Excel
pygmt             # (opcional) visualização topográfica avançada
```

Instale tudo com:
```bash
pip install earthengine-api rasterio geopandas scikit-learn scikit-image pandas numpy matplotlib contextily joblib openpyxl pygmt
```

---

## Como Usar

**1. Preparar as máscaras de água:**
```bash
python preparar_dados.py
```
Autentique-se no GEE quando solicitado. As tarefas de exportação ficam visíveis no painel do Earth Engine.

**2. Treinar o modelo e gerar previsões:**
```bash
python iaparaprevisao.py
```
Ajuste os caminhos dos dados no topo do arquivo antes de executar. Se um modelo já treinado existir em disco, ele será carregado automaticamente.

---

## Área de Estudo

| Campo | Valor |
|---|---|
| Município | Juazeiro do Norte – CE, Brasil |
| Coordenadas | −7.2126° S, −39.3136° O |
| Raio de análise | 15 km |
| Resolução do grid | 10 m |
| Satélite | Sentinel-1 C-SAR (modo IW, polarização VH) |
| DEM | ANADEM v1 (~24 m) |
| LULC | Dynamic World (Google Earth Engine) |
