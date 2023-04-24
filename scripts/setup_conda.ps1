# Default RECIPE 'rome' can be overridden by 'RECIPE' environment variable
$RECIPE = if ($env:RECIPE) { $env:RECIPE }{ "rome" }
# Default ENV_NAME 'rome' can be overridden by 'ENV_NAME'
$ENV_NAME = if ($env:ENV_NAME) { $env:ENV_NAME } else { $env:RECIPE }

Write-Host "Creating conda environment $ENV_NAME"

if (!(Get-Command conda -ErrorAction SilentlyContinue)) {
    Write-Host "conda not in PATH"
    Write-Host "read: https://conda.io/docs/user-guide/install/index.html"
    exit 1
}

if ((Get-Item "${HOME}.conda" -ErrorAction SilentlyContinue).Attributes -match "ReparsePoint") {
    Write-Host "Not installing: your ~/.conda directory is on AFS."
    Write-Host "Use 'mklink /d C:\some\nfs\dir ${HOME}.conda' to avoid using up your AFS quota."
    exit 1
}

$CUDA_DIR = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7"
if (!(Test-Path $CUDA_DIR -PathType Container)) {
    Write-Host "Environment requires $CUDA_DIR, not found."
    exit 1
}

# Uninstall existing environment
# conda deactivate
conda env remove --name $ENV_NAME

# Build new environment: torch and torch vision from source
# CUDA_HOME is needed
# https://github.com/rusty1s/pytorch_scatter/issues/19#issuecomment-449735614
conda env create --name $ENV_NAME -f "$RECIPE.yml"

# Set up CUDA_HOME to set itself up correctly on every source activate
# https://stackoverflow.com/questions/31598963
New-Item -ItemType Directory -Path "$env:HOME.conda\envs$ENV_NAME\etc\conda\activate.d" -Force | Out-Null
$env:CUDA_HOME = $CUDA_DIR
"export CUDA_HOME=$env:CUDA_HOME" | Out-File -Encoding ascii "$env:HOME.conda\envs$ENV_NAME\etc\conda\activate.d\CUDA_HOME.ps1"

# conda activate ${ENV_NAME}
