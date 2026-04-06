#!/bin/bash

# ==============================================================================
# SDSS QML PIPELINE AUTOMATOR
# ==============================================================================
# This script manages the full data lifecycle: Download -> Precheck -> Extract.
# ==============================================================================

# Exit immediately if a command exits with a non-zero status
set -e

# Paths
DATASET_DIR="dataset"
ASSETS_DIR="$DATASET_DIR/assets"
CATALOG_FILE="spAll-lite-v6_1_3.fits.gz"
CATALOG_PATH="$ASSETS_DIR/$CATALOG_FILE"
CATALOG_URL="https://data.sdss.org/sas/dr19/spectro/boss/redux/v6_1_3/$CATALOG_FILE"

# Colors for fancy
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}Starting SDSS QML Pipeline...${NC}"

# --- PHASE 0: PREPARE CATALOG ---
if [ ! -f "$CATALOG_PATH" ]; then
    echo -e "${YELLOW}Catalog $CATALOG_FILE not found in $ASSETS_DIR.${NC}"
    echo -e "${BLUE}Downloading 1.5GB catalog (this may take a few minutes)...${NC}"
    mkdir -p "$ASSETS_DIR"
    curl -L "$CATALOG_URL" -o "$CATALOG_PATH"
else
    echo -e "${GREEN}Found catalog at $CATALOG_PATH${NC}"
fi

# --- MOVE INTO WORKSPACE ---
cd "$DATASET_DIR"

# --- PHASE 1: SAMPLES PRECHECK ---
echo -e "\n${BLUE}[Phase 1/3] Running Samples Pre-check...${NC}"
uv run samples_precheck.py

# --- PHASE 2: DOWNLOAD SPECTRA ---
echo -e "\n${BLUE}[Phase 2/3] Downloading Selected SDSS Spectra...${NC}"
# We run the download script. It uses your TARGET_CATEGORIES and SAMPLES_PER_SUBCLASS.
uv run download_sdss_data.py

# --- PHASE 3: BUILD DATASET ---
echo -e "\n${BLUE}[Phase 3/3] Extracting FITS and Building Parquet...${NC}"
# This merges subclasses and packages everything into ML_Training_Data.parquet
uv run build_ml_dataset.py

echo -e "\n${GREEN}PIPELINE COMPLETE! ✨${NC}"
echo -e "Your training data is ready at: ${DATASET_DIR}/ML_Training_Data.parquet"