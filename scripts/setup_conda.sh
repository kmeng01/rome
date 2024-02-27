#!/bin/bash

# Start from directory of script
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR

# Detect operating system
unameOut="$(uname -s)"
case "${unameOut}" in
    Linux*)     machine=Linux;;
    Darwin*)    machine=Mac;;
    CYGWIN*)    machine=Cygwin;;
    MINGW*)     machine=MinGw;;
    *)          machine="UNKNOWN:${unameOut}"
esac

if [ $machine != "Linux" ] && [ $machine != "Mac" ]
then
	echo "Conda setup script is only available on Linux and Mac."
	exit 1
else
	echo "Running on $machine..."
fi

if [[ -z "${CONDA_HOME}" ]]; then
  echo "Please specify the CONDA_HOME environment variable (it might look something like ~/miniconda3)."
  exit 1
else
  echo "Found CONDA_HOME=${CONDA_HOME}."
fi

RECIPE=${RECIPE:-rome_new}
ENV_NAME="${ENV_NAME:-${RECIPE}}"
echo "Creating conda environment ${ENV_NAME}..."

if [[ ! $(type -P conda) ]]
then
    echo "conda not in PATH"
    echo "read: https://conda.io/docs/user-guide/install/index.html"
    exit 1
fi

if df "${HOME}/.conda" --type=afs > /dev/null 2>&1
then
    echo "Not installing: your ~/.conda directory is on AFS."
    echo "Use 'ln -s /some/nfs/dir ~/.conda' to avoid using up your AFS quota."
    exit 1
fi

# Build new environment
conda env create --name=${ENV_NAME} -f ${RECIPE}.yml
