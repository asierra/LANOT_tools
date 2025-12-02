#!/bin/bash
#
# Script de desinstalación de LANOT_tools
#
# Uso: sudo ./uninstall.sh

set -e

# Colores
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Verificar root
if [ "$EUID" -ne 0 ]; then 
    echo -e "${RED}Error: Este script debe ejecutarse como root (use sudo)${NC}"
    exit 1
fi

INSTALL_DIR="/opt/lanot-tools"
BIN_WRAPPER="/usr/local/bin/mapdrawer"

echo -e "${YELLOW}=== Desinstalación de LANOT_tools ===${NC}"
echo ""

# Confirmar
read -p "¿Está seguro de desinstalar LANOT_tools? (s/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Ss]$ ]]; then
    echo "Desinstalación cancelada."
    exit 0
fi

# Eliminar comando
if [ -f "${BIN_WRAPPER}" ]; then
    rm -f "${BIN_WRAPPER}"
    echo -e "${GREEN}✓ Eliminado: ${BIN_WRAPPER}${NC}"
fi

# Eliminar directorio de instalación
if [ -d "${INSTALL_DIR}" ]; then
    rm -rf "${INSTALL_DIR}"
    echo -e "${GREEN}✓ Eliminado: ${INSTALL_DIR}${NC}"
fi

echo ""
echo -e "${GREEN}Desinstalación completada.${NC}"
