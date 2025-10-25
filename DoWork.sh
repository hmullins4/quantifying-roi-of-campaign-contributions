#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

# Detect whether sudo is usable; if not, fall back to no-sudo mode
if sudo -n true 2>/dev/null; then
  SUDO="sudo"
else
  echo "→ Warning: no sudo privileges detected; skipping all system‑package installs!" >&2
  SUDO=""
fi

# Get the absolute path of the directory containing this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Quiet mode: send only stdout into the log; keep stderr (our echos) on the console
LOGFILE="$SCRIPT_DIR/install.log"
exec 1>"$LOGFILE"

echo "Installation log is being written to $LOGFILE." >&2

# Trap errors for easier debugging
trap 'echo "Error on line $LINENO: $BASH_COMMAND" >&2' ERR

# Detect package manager
if   command -v apt-get >/dev/null 2>&1; then PKG_MGR='apt-get';
elif command -v yum     >/dev/null 2>&1; then PKG_MGR='yum';
elif command -v dnf     >/dev/null 2>&1; then PKG_MGR='dnf';
elif command -v brew    >/dev/null 2>&1; then PKG_MGR='brew';
else echo "No supported package manager (apt-get, yum, dnf, brew) found." >&2; exit 1; fi

# Detect best available python3 (version >= 3.9)
PYTHON=""
for candidate in python3.13 python3.12 python3.11 python3.10 python3.9 python3; do
  if command -v "$candidate" >/dev/null 2>&1; then
    if "$candidate" -c "import sys; sys.exit(0) if sys.version_info >= (3,9) else sys.exit(1)"; then
      PYTHON="$candidate"
      break
    fi
  fi
done

if [ -z "$PYTHON" ]; then
  echo "ERROR: No suitable Python 3.9+ interpreter found. Please install Python 3.9 or newer." >&2
  exit 1
fi

echo "→ Using Python interpreter: $PYTHON" >&2

# ────────────────────────────────────────────────────────────────────
# Create & activate a dedicated venv so all pip installs go inside it
echo "→ (Re)creating Python venv for package installs…" >&2
rm -rf "$SCRIPT_DIR/venv"
"$PYTHON" -m venv "$SCRIPT_DIR/venv"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/venv/bin/activate"
echo "→ Activated venv: $SCRIPT_DIR/venv" >&2
echo "→ Upgrading pip in venv…" >&2
pip install --upgrade pip setuptools wheel --quiet
# ────────────────────────────────────────────────────────────────────

# Function to install system tools if missing
install_if_missing() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "→ '$cmd' not found; installing full LaTeX environment..." >&2
    case "$PKG_MGR" in
      apt-get)
        sudo apt-get update -qq
        sudo apt-get install -y -qq texlive-full
        ;;
      yum)
        sudo yum install -y -q texlive-scheme-full
        ;;
      dnf)
        sudo dnf install -y -q texlive-scheme-full
        ;;
      brew)
        brew install --cask mactex-no-gui >/dev/null
        ;;
    esac
    echo "→ '$cmd' installed." >&2
  else
    echo "→ '$cmd' already installed." >&2
  fi
}

# Function to install pip3 if missing
install_pip_if_missing() {
  if ! "$PYTHON" -m pip --version >/dev/null 2>&1; then
    echo "→ pip not found for $PYTHON; installing pip..." >&2
    case "$PKG_MGR" in
      apt-get)
        $SUDO apt-get update -qq
        $SUDO apt-get install -y -qq python3-pip
        ;;
      yum)
        $SUDO yum install -y -q python3-pip
        ;;
      dnf)
        $SUDO dnf install -y -q python3-pip
        ;;
      brew)
        echo "Please install pip manually on macOS if missing." >&2
        ;;
    esac
    echo "→ pip installed." >&2
  else
    echo "→ pip already installed for $PYTHON." >&2
  fi
}

# Function to install Python packages if missing
install_python_package_if_missing() {
  local module_name="$1"
  local pip_name="${2:-$1}"

  if ! "$PYTHON" -c "import $module_name" &> /dev/null; then
    echo "→ Python package '$pip_name' not found; installing via pip..." >&2
    "$PYTHON" -m pip install --quiet "$pip_name" || {
      echo "→ pip install failed; falling back to system package via $PKG_MGR..." >&2
      case "$PKG_MGR" in
        apt-get)
          $SUDO apt-get update -qq
          $SUDO apt-get install -y -qq "python3-${pip_name}" 2>>"$LOGFILE"
          ;;
        yum)
          $SUDO yum install -y -q "python3-${pip_name}" 2>>"$LOGFILE"
          ;;
        dnf)
          $SUDO dnf install -y -q "python3-${pip_name}" 2>>"$LOGFILE"
          ;;
        brew)
          brew install "$pip_name" >/dev/null
          ;;
      esac
    }
    echo "→ Python package '$pip_name' installed." >&2
  else
    echo "→ Python package '$pip_name' already present." >&2
  fi
}

# Function to ensure python3-venv is available
install_venv_package() {
  echo "→ Checking for python3 venv + ensurepip support…" >&2
  if "$PYTHON" -c "import ensurepip" 2>/dev/null; then
    echo "→ python3-venv with ensurepip already present." >&2
    return 0
  fi
  echo "→ ensurepip missing; installing python3-venv packages…" >&2
  VENV_PKG="$($PYTHON -c 'import sys; print(f"python{sys.version_info.major}.{sys.version_info.minor}-venv")')"
  . /etc/os-release 2>/dev/null || { echo "Cannot detect OS; please install ${VENV_PKG} manually." >&2; return 1; }
  case "$ID" in
    ubuntu|debian)
      $SUDO apt-get update -qq
      $SUDO apt-get install -y -qq python3-venv "$VENV_PKG"
      ;;
    fedora)
      $SUDO dnf install -y -q python3-venv
      ;;
    centos|rhel)
      $SUDO yum install -y -q python3-venv
      ;;
    opensuse*|suse)
      $SUDO zypper install -y -q python3-venv
      ;;
    *)
      echo "Unsupported distro: $ID. Install ${VENV_PKG} manually." >&2
      return 1
      ;;
  esac
  echo "→ python3-venv (including $VENV_PKG) installed." >&2
}

echo "Running from script directory: $SCRIPT_DIR" >&2
echo "===========================" >&2
echo " Checking for missing LaTeX and Python packages." >&2
echo "===========================" >&2

install_if_missing pdflatex
install_if_missing bibtex
install_if_missing "$PYTHON"

install_pip_if_missing

for pair in \
    "matplotlib" \
    "numpy" \
    "pandas" \
    "pygam" \
    "sklearn scikit-learn" \
    "duckdb" \
    "statsmodels" \
    "seaborn" \
    "pyarrow" \
    "jinja2"
do
  IFS=' ' read -r module_name pip_name <<< "$pair"
  install_python_package_if_missing "$module_name" "${pip_name:-$module_name}"
done

install_venv_package

# Install C build deps for matplotlib
echo "→ Installing build deps…" >&2
if [ -n "$SUDO" ]; then
  case "$PKG_MGR" in
    apt-get)
      $SUDO apt-get install -y -qq libfreetype6-dev pkg-config libpng-dev python3-dev 2>>"$LOGFILE"
      ;;
    yum)
      $SUDO yum install -y -q freetype-devel libpng-devel python3-devel pkgconfig 2>>"$LOGFILE"
      ;;
    dnf)
      $SUDO dnf install -y -q freetype-devel libpng-devel python3-devel pkgconfig 2>>"$LOGFILE"
      ;;
    brew)
      brew install freetype libpng pkg-config >/dev/null
      ;;
  esac
  echo "→ Build deps done." >&2
else
  echo "→ No sudo: skipping system build‑deps. Your plots may fail if those libs aren’t already present." >&2
fi

# Install requirements from file
echo "→ Installing requirements.txt…" >&2
if ! pip install -r requirements.txt --quiet; then
  echo "→ requirements.txt install failed; installing system matplotlib..." >&2
  case "$PKG_MGR" in
    apt-get) $SUDO apt-get install -y -qq python3-matplotlib 2>>"$LOGFILE";;
    yum)     $SUDO yum install -y -q python3-matplotlib 2>>"$LOGFILE";;
    dnf)     $SUDO dnf install -y -q python3-matplotlib 2>>"$LOGFILE";;
    brew)    brew install matplotlib >/dev/null;;
  esac
fi
echo "→ Python requirements installed." >&2

echo "→ Installation complete." >&2

echo >&2
echo "===========================" >&2
echo " Cleaning up intermediate files..." >&2
echo "===========================" >&2
rm -f "$SCRIPT_DIR"/MyPapers/HEMullins-Paper.{aux,log,out,bbl,blg,toc,lof,lot}

echo >&2
echo "===========================" >&2
echo " Running Python code to generate plots and tables." >&2
echo "===========================" >&2

echo "→ Running dist.py…" >&2
"$PYTHON" "$SCRIPT_DIR/Code/dist.py" > "$SCRIPT_DIR/Figures/plot.log" 2>&1 || {
  echo "ERROR: dist.py failed. See Figures/plot.log for details." >&2
  exit 1
}
echo "===========================" >&2
echo "Plot generated: $SCRIPT_DIR/Figures/amount_dist.pdf" >&2
echo "===========================" >&2

echo "→ Running spending_trend.py…" >&2
"$PYTHON" "$SCRIPT_DIR/Code/spending_trend.py" > "$SCRIPT_DIR/Figures/plot.log" 2>&1 || {
  echo "ERROR: spending_trend.py failed. See Figures/plot.log for details." >&2
  exit 1
}
echo "===========================" >&2
echo "Plot generated: $SCRIPT_DIR/Figures/cycle_spending_trend.pdf" >&2
echo "===========================" >&2

echo "→ Running roi_over_time_visual.py…" >&2
"$PYTHON" "$SCRIPT_DIR/Code/roi_over_time_visual.py" > "$SCRIPT_DIR/Figures/plot.log" 2>&1 || {
  echo "ERROR: roi_over_time_visual.py failed. See Figures/plot.log for details." >&2
  exit 1
}
echo "===========================" >&2
echo "Plot generated: $SCRIPT_DIR/Figures/roi_over_time.pdf" >&2
echo "Plot generated: $SCRIPT_DIR/Figures/roi_beta_over_time.pdf" >&2
echo "Plot generated: $SCRIPT_DIR/elasticity_over_time.pdf" >&2
echo "Plot generated: $SCRIPT_DIR/norm_elast_over_time.pdf" >&2
echo "===========================" >&2

echo "→ Running gam_plots.py…" >&2
"$PYTHON" "$SCRIPT_DIR/Code/gam_plots.py" > "$SCRIPT_DIR/Figures/plot.log" 2>&1 || {
  echo "ERROR: gam_plots.py failed. See Figures/plot.log for details." >&2
  exit 1
}
echo "===========================" >&2
echo "Plot generated: $SCRIPT_DIR/Figures/gam_roi_over_time.pdf" >&2
echo "Plot generated: $SCRIPT_DIR/Figures/gam_roi_beta_over_time.pdf" >&2
echo "Plot generated: $SCRIPT_DIR/Figures/gam_elasticity_over_time.pdf" >&2
echo "Plot generated: $SCRIPT_DIR/Figures/gam_norm_elast_over_time.pdf" >&2
echo "===========================" >&2

echo "→ Running model_linear.py…" >&2
"$PYTHON" "$SCRIPT_DIR/Code/model_linear.py" > "$SCRIPT_DIR/Figures/plot.log" 2>&1 || {
  echo "ERROR: model_linear.py failed. See Figures/plot.log for details." >&2
  exit 1
}
echo "===========================" >&2
echo "Plot generated: $SCRIPT_DIR/Figures/feat_imp.pdf" >&2
echo "Plot generated: $SCRIPT_DIR/Figures/sig_heatmap.pdf" >&2
echo "Table generated: $SCRIPT_DIR/Tables/lin_metrics.tex" >&2
echo "Table generated: $SCRIPT_DIR/Tables/lin_roi.tex" >&2
echo "Table generated: $SCRIPT_DIR/Tables/lin_e.tex" >&2
echo "Table generated: $SCRIPT_DIR/Tables/reg_r2.tex" >&2
echo "===========================" >&2

echo "→ Running model_logistic.py…" >&2
"$PYTHON" "$SCRIPT_DIR/Code/model_logistic.py" > "$SCRIPT_DIR/Figures/plot.log" 2>&1 || {
  echo "ERROR: model_logistic.py failed. See Figures/plot.log for details." >&2
  exit 1
}
echo "===========================" >&2
echo "Plot generated: $SCRIPT_DIR/Figures/log_roc.pdf" >&2
echo "Table generated: $SCRIPT_DIR/Tables/log_e.tex" >&2
echo "Table generated: $SCRIPT_DIR/Tables/log_metrics.tex" >&2
echo "Table generated: $SCRIPT_DIR/Tables/log_roi.tex" >&2
echo "Table generated: $SCRIPT_DIR/Tables/log_type_roi.tex" >&2
echo "===========================" >&2

echo "→ Running model_logisticGAM.py…" >&2
"$PYTHON" "$SCRIPT_DIR/Code/model_logisticGAM.py" > "$SCRIPT_DIR/Figures/plot.log" 2>&1 || {
  echo "ERROR: model_logisticGAM.py failed. See Figures/plot.log for details." >&2
  exit 1
}
echo "===========================" >&2
echo "Plot generated: $SCRIPT_DIR/Figures/gam_log_roc.pdf" >&2
echo "Table generated: $SCRIPT_DIR/Tables/log_gam_e.tex" >&2
echo "Table generated: $SCRIPT_DIR/Tables/log_gam_metrics.tex" >&2
echo "Table generated: $SCRIPT_DIR/Tables/log_gam_roi.tex" >&2
echo "Table generated: $SCRIPT_DIR/Tables/log_gam_type_roi.tex" >&2
echo "===========================" >&2

echo "→ Running model_linearGAM.py…" >&2
"$PYTHON" "$SCRIPT_DIR/Code/model_linearGAM.py" > "$SCRIPT_DIR/Figures/plot.log" 2>&1 || {
  echo "ERROR: model_linearGAM.py failed. See Figures/plot.log for details." >&2
  exit 1
}
echo "===========================" >&2
echo "Table generated: $SCRIPT_DIR/Tables/lin_gam_cv.tex" >&2
echo "Table generated: $SCRIPT_DIR/Tables/lin_gam_e.tex" >&2
echo "Table generated: $SCRIPT_DIR/Tables/lin_gam_ho.tex" >&2
echo "===========================" >&2

echo "===========================" >&2
echo " Compiling HEMullins-Paper.tex..." >&2
echo "===========================" >&2

pushd "$SCRIPT_DIR"/MyPapers > /dev/null
pdflatex -interaction=nonstopmode HEMullins-Paper.tex > pdflatex1.log 2>&1
bibtex HEMullins-Paper > bibtex.log 2>&1
pdflatex -interaction=nonstopmode HEMullins-Paper.tex > pdflatex2.log 2>&1
pdflatex -interaction=nonstopmode HEMullins-Paper.tex > pdflatex3.log 2>&1
popd > /dev/null

echo "===========================" >&2
echo " Done. Hallelujah." >&2
echo "===========================" >&2
echo "Final PDF: $SCRIPT_DIR/MyPapers/HEMullins-Paper.pdf" >&2
