#!/bin/bash

# Clear existing data
echo "Cleaning up existing data..."
rm -rf src/imgDB
rm -rf src/features/image_features/ViT-21k
rm -rf src/features/text_features
echo "Cleanup complete."


# ========================================= Database =========================================
# Ajout de la base de données d'images
mkdir -p src/imgDB
echo "Downloading image database..."
wget -q --show-progress -O src/images.zip https://github.com/sidimahmoudi/facenet_tf2/releases/download/AI_MIR_CLOUD/MIR_DATASETS_B.zip


# Décompression de l'archive
echo "Unzipping image database..."
unzip -q src/images.zip -d src/
mv src/MIR_DATASETS_B/* src/imgDB/
rm -r src/MIR_DATASETS_B
# Suppression de l'archive zip
rm src/images.zip


# ========================================= Features =========================================

# Setup du dossier de features
mkdir -p src/features
mkdir -p src/features/image_features


# Telechargement des features de texte
echo "Downloading text features..."
wget -q --show-progress -O src/features/text_features.zip https://github.com/MrRose765/MIR_2025/releases/download/features_v1.0.0/text_features.zip

echo "Unzipping text features..."
unzip -q src/features/text_features.zip -d src/features
rm src/features/text_features.zip

# Telechargement des features d'images générées sur Google Colab
echo "Downloading image features..."

# 1 - ViT-21k
wget -q --show-progress -O src/features/image_features/ViT-21k.zip https://github.com/MrRose765/MIR_2025/releases/download/features_v1.0.0/ViT-21k.zip
unzip -q src/features/image_features/ViT-21k.zip -d src/features/image_features
rm src/features/image_features/ViT-21k.zip

# ...