#!/usr/bin/env bash
set -euo pipefail

export ANDROID_SDK_ROOT="$HOME/Android/Sdk"
export PATH="$ANDROID_SDK_ROOT/platform-tools:$PATH"
export PATH="$ANDROID_SDK_ROOT/cmdline-tools/latest/bin:$PATH"

# Determine repository root relative to script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
PROJECT_PATH="$REPO_ROOT/PoopDetector/PoopDetector.csproj"
OUTPUT_DIR="$REPO_ROOT/PoopDetector/bin/Release/net9.0-android/publish"

command_exists() {
    command -v "$1" >/dev/null 2>&1
}

require_command() {
    if ! command_exists "$1"; then
        echo "Error: Required command '$1' is not available in PATH." >&2
        exit 1
    fi
}

check_dotnet_sdk() {
    if ! dotnet --list-sdks | grep -q '^9\.'; then
        echo "Error: .NET 9 SDK (preview) is required. Install it from https://dotnet.microsoft.com/download/dotnet/9.0." >&2
        exit 1
    fi
}

ensure_maui_workload() {
    if ! dotnet workload list | grep -q '^maui'; then
        echo "Installing required 'maui' workload..."
        dotnet workload install maui-android android
    fi
}

check_android_dependencies() {
    require_command "java"
    require_command "adb"
}

ensure_android_sdk_manager() {
    if command_exists "sdkmanager"; then
        local required_packages=("platform-tools" "platforms;android-34" "build-tools;34.0.0")
        for pkg in "${required_packages[@]}"; do
            if ! sdkmanager --list | grep -q "$pkg"; then
                echo "Installing Android SDK package: $pkg"
                yes | sdkmanager "$pkg"
            fi
        done
    else
        echo "Warning: 'sdkmanager' not found; skipping Android SDK package verification." >&2
    fi
}

build_apk() {
    echo "Restoring .NET dependencies..."
    echo "PROJECT_PATH = $PROJECT_PATH"
    dotnet restore "$PROJECT_PATH"

    echo "Building Release APK..."
    dotnet publish "$PROJECT_PATH" -f net9.0-android -c Release

    dotnet restore PoopDetector.sln -p:TargetFramework=net9.0-android
    dotnet build   PoopDetector.sln -p:TargetFramework=net9.0-android
}

find_apk() {
    if [ ! -d "$OUTPUT_DIR" ]; then
        echo "Error: Publish output directory not found: $OUTPUT_DIR" >&2
        exit 1
    fi

    local apk
    apk=$(find "$OUTPUT_DIR" -maxdepth 1 -name '*.apk' | head -n 1)
    if [ -z "$apk" ]; then
        echo "Error: No APK found in $OUTPUT_DIR" >&2
        exit 1
    fi
    echo "$apk"
}

ensure_device_connected() {
    echo "Checking for connected Android devices..."
    local devices
    devices=$(adb devices | awk 'NR>1 && $2=="device" {print $1}')
    if [ -z "$devices" ]; then
        echo "Error: No connected Android devices detected via 'adb'. Ensure USB debugging is enabled." >&2
        adb devices
        exit 1
    fi
    printf "Detected devices:\n%s" "$devices"
}

install_apk() {
    local apk_path="$1"
    echo "Installing APK: $apk_path"
    adb install -r "$apk_path"
}

main() {
    require_command "dotnet"
    check_dotnet_sdk
    ensure_maui_workload
    check_android_dependencies
    ensure_android_sdk_manager

    build_apk
    local apk_path
    apk_path=$(find_apk)

    ensure_device_connected
    install_apk "$apk_path"

    echo "APK installation complete."
}

main "$@"

