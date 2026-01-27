#!/bin/bash
#
# Script de instalación de LANOT_tools para servidor
# Instala el paquete en /opt/lanot-tools con virtualenv
# y crea comando mapdrawer accesible globalmente
#
# Uso: sudo ./install.sh

set -e  # Salir si hay errores

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Verificar que se ejecuta como root
if [ "$EUID" -ne 0 ]; then 
    echo -e "${RED}Error: Este script debe ejecutarse como root (use sudo)${NC}"
    exit 1
fi

# Configuración
INSTALL_DIR="/opt/lanot-tools"
VENV_DIR="${INSTALL_DIR}/venv"
SRC_DIR="${INSTALL_DIR}/src"
BIN_WRAPPER_MD="/usr/local/bin/mapdrawer"
BIN_WRAPPER_G2V="/usr/local/bin/geotiff2view"
SHARE_DIR="/usr/local/share/lanot"
CPT_DIR="${SHARE_DIR}/colortables"

# Directorio del script (donde está el código fuente)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo -e "${GREEN}=== Instalación de LANOT_tools ===${NC}"
echo ""
echo "Directorio de instalación: ${INSTALL_DIR}"
echo "Código fuente desde: ${SCRIPT_DIR}"
echo ""

# Paso 1: Verificar dependencias del sistema
echo -e "${YELLOW}[1/7] Verificando dependencias del sistema...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: python3 no está instalado${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
echo "  ✓ Python ${PYTHON_VERSION} encontrado"

# Paso 2: Crear directorio de instalación
echo -e "${YELLOW}[2/7] Creando directorio de instalación...${NC}"
mkdir -p "${INSTALL_DIR}"
echo "  ✓ Directorio ${INSTALL_DIR} creado"

# Paso 3: Copiar código fuente
echo -e "${YELLOW}[3/7] Copiando código fuente...${NC}"
rm -rf "${SRC_DIR}"
cp -r "${SCRIPT_DIR}" "${SRC_DIR}"
# Limpiar archivos innecesarios
rm -rf "${SRC_DIR}/.git" "${SRC_DIR}/__pycache__" "${SRC_DIR}"/*.pyc "${SRC_DIR}"/build "${SRC_DIR}"/*.egg-info
echo "  ✓ Código fuente copiado a ${SRC_DIR}"

# Paso 4: Instalar recursos (CPTs)
echo -e "${YELLOW}[4/7] Instalando recursos compartidos...${NC}"
mkdir -p "${CPT_DIR}"
if ls "${SCRIPT_DIR}"/*.cpt >/dev/null 2>&1; then
    cp "${SCRIPT_DIR}"/*.cpt "${CPT_DIR}/"
    echo "  ✓ Tablas de color (.cpt) copiadas a ${CPT_DIR}"
elif [ -d "${SCRIPT_DIR}/colortables" ]; then
    cp -r "${SCRIPT_DIR}/colortables/"* "${CPT_DIR}/"
    echo "  ✓ Tablas de color copiadas desde colortables/ a ${CPT_DIR}"
else
    echo "  - No se encontraron archivos .cpt en origen, omitiendo copia."
fi

# Paso 5: Crear/actualizar virtualenv
echo -e "${YELLOW}[5/7] Configurando virtualenv...${NC}"
if [ -d "${VENV_DIR}" ]; then
    echo "  - Virtualenv existente encontrado, recreando..."
    rm -rf "${VENV_DIR}"
fi
python3 -m venv "${VENV_DIR}"
echo "  ✓ Virtualenv creado en ${VENV_DIR}"

# Paso 6: Instalar paquete
echo -e "${YELLOW}[6/7] Instalando paquete y dependencias...${NC}"
echo "  - Actualizando pip..."
"${VENV_DIR}/bin/pip" install --upgrade pip --quiet
echo "  ✓ pip actualizado"
echo "  - Instalando lanot-tools (esto puede tomar un momento)..."
if "${VENV_DIR}/bin/pip" install "${SRC_DIR}" --quiet; then
    echo "  ✓ lanot-tools instalado con dependencias"
else
    echo -e "${RED}  ✗ Error instalando lanot-tools${NC}"
    exit 1
fi

# Paso 7: Crear wrapper script
echo -e "${YELLOW}[7/7] Creando comandos globales...${NC}"
cat > "${BIN_WRAPPER_MD}" << 'EOF'
#!/bin/bash
# Wrapper para mapdrawer - ejecuta desde virtualenv
VENV_DIR="/opt/lanot-tools/venv"
exec "${VENV_DIR}/bin/mapdrawer" "$@"
EOF

chmod +x "${BIN_WRAPPER_MD}"
echo "  ✓ Comando ${BIN_WRAPPER_MD} creado"

cat > "${BIN_WRAPPER_G2V}" << 'EOF'
#!/bin/bash
# Wrapper para geotiff2view - ejecuta desde virtualenv
VENV_DIR="/opt/lanot-tools/venv"
exec "${VENV_DIR}/bin/geotiff2view" "$@"
EOF

chmod +x "${BIN_WRAPPER_G2V}"
echo "  ✓ Comando ${BIN_WRAPPER_G2V} creado"

# Verificación
echo ""
echo -e "${YELLOW}Verificando instalación...${NC}"
if command -v mapdrawer &> /dev/null && mapdrawer --help > /dev/null 2>&1 && \
   command -v geotiff2view &> /dev/null && geotiff2view --help > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Instalación completada exitosamente!${NC}"
    echo ""
    echo "Puede usar los comandos desde cualquier ubicación:"
    echo "  $ mapdrawer --help"
    echo "  $ geotiff2view --help"
    echo ""
    echo "Para desinstalar, ejecute:"
    echo "  $ sudo ${SCRIPT_DIR}/uninstall.sh"
    echo "  o manualmente: sudo rm -rf ${INSTALL_DIR} ${BIN_WRAPPER_MD} ${BIN_WRAPPER_G2V}"
else
    echo -e "${RED}✗ Error en la verificación. La instalación puede estar incompleta.${NC}"
    echo "Intente ejecutar manualmente: ${BIN_WRAPPER_MD} --help y ${BIN_WRAPPER_G2V} --help"
    exit 1
fi
