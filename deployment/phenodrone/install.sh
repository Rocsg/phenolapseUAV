#!/usr/bin/env bash
# Install one centrally managed PhenoLapseUAV environment on a Linux workstation/server.
# Run as root: ./install.sh <git-repository-url> [git-branch]
set -euo pipefail

if [[ ${EUID} -ne 0 ]]; then
    echo "Run this installer as root (for example via sudo)." >&2
    exit 1
fi
if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: $0 <git-repository-url> [git-branch]" >&2
    exit 2
fi

readonly REPOSITORY_URL=$1
readonly BRANCH=${2:-main}
readonly INSTALL_ROOT=/opt/phenodrone
readonly REPOSITORY_DIRECTORY="$INSTALL_ROOT/phenolapseUAV"
readonly VENV_DIRECTORY="$INSTALL_ROOT/venv"
readonly LAUNCHER_DIRECTORY="$INSTALL_ROOT/bin"
readonly APPLICATION_DIRECTORY=/usr/share/applications
readonly PYTHON_BIN=${PYTHON_BIN:-python3.11}

command -v "$PYTHON_BIN" >/dev/null || {
    echo "Python 3.11 or newer is required; set PYTHON_BIN if necessary." >&2
    exit 1
}
"$PYTHON_BIN" -c 'import sys; raise SystemExit(sys.version_info < (3, 11))' || {
    echo "Python 3.11 or newer is required." >&2
    exit 1
}

install -d -m 0755 "$INSTALL_ROOT" "$LAUNCHER_DIRECTORY"
if [[ -d "$REPOSITORY_DIRECTORY/.git" ]]; then
    git -C "$REPOSITORY_DIRECTORY" fetch origin "$BRANCH"
    git -C "$REPOSITORY_DIRECTORY" checkout "$BRANCH"
    git -C "$REPOSITORY_DIRECTORY" pull --ff-only origin "$BRANCH"
else
    git clone --branch "$BRANCH" "$REPOSITORY_URL" "$REPOSITORY_DIRECTORY"
fi

if [[ ! -x "$VENV_DIRECTORY/bin/python" ]]; then
    "$PYTHON_BIN" -m venv "$VENV_DIRECTORY"
fi
"$VENV_DIRECTORY/bin/python" -m pip install --upgrade pip
"$VENV_DIRECTORY/bin/python" -m pip install --upgrade "$REPOSITORY_DIRECTORY"
"$VENV_DIRECTORY/bin/python" -m pip check

cat > "$LAUNCHER_DIRECTORY/phenolapseuav" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
# Safer over X11 forwarding; avoids shared-memory errors with remote displays.
export QT_X11_NO_MITSHM=1
exec /opt/phenodrone/venv/bin/phenolapseUAV-gui "$@"
EOF
chmod 0755 "$LAUNCHER_DIRECTORY/phenolapseuav"

cat > "$LAUNCHER_DIRECTORY/phenolapseuav-update" <<'EOF'
#!/usr/bin/env bash
# Run only when no colleague is actively using PhenoLapseUAV.
set -euo pipefail
readonly ROOT=/opt/phenodrone
readonly REPOSITORY="$ROOT/phenolapseUAV"
readonly PYTHON="$ROOT/venv/bin/python"
readonly LOCKFILE="$ROOT/.update.lock"

exec 9>"$LOCKFILE"
flock -n 9 || {
    echo "Another PhenoLapseUAV update is already running." >&2
    exit 1
}
git -C "$REPOSITORY" diff --quiet || {
    echo "Refusing to overwrite uncommitted central changes." >&2
    exit 1
}
git -C "$REPOSITORY" pull --ff-only
"$PYTHON" -m pip install --upgrade "$REPOSITORY"
"$PYTHON" -m pip check
printf 'PhenoLapseUAV updated to %s.\n' "$(git -C "$REPOSITORY" rev-parse --short HEAD)"
EOF
chmod 0755 "$LAUNCHER_DIRECTORY/phenolapseuav-update"

cat > "$APPLICATION_DIRECTORY/phenolapseuav.desktop" <<'EOF'
[Desktop Entry]
Type=Application
Name=PhenoLapseUAV
Comment=UAV orthomosaic registration in napari
Exec=/opt/phenodrone/bin/phenolapseuav
Terminal=false
Categories=Science;Geography;
StartupNotify=true
EOF
chmod 0644 "$APPLICATION_DIRECTORY/phenolapseuav.desktop"

printf 'Installed PhenoLapseUAV in %s.\n' "$INSTALL_ROOT"
printf 'All users can now start it from the applications menu or with: %s\n' \
    "$LAUNCHER_DIRECTORY/phenolapseuav"
printf 'An administrator updates it with: sudo %s\n' \
    "$LAUNCHER_DIRECTORY/phenolapseuav-update"
