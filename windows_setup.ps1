# In VSCode I also installed the recommended C# dev kit, not sure if that matters.

# If using a windows VM
<#

VM_NAME=WinDev2407Eval
VBoxManage startvm "$VM_NAME"

ssh -p 2222 user@localhost

cd $HOME/code/shitspotter/tpl/poopdetector

#>





# Open PowerShell (Admin) and run:

# .NET SDK 9 (required for .NET MAUI 9)
winget install --id Microsoft.DotNet.SDK.9 -e

# Microsoft OpenJDK 17 (the MAUI toolchain expects a JDK; 17 is the current LTS)
winget install --id Microsoft.OpenJDK.17 -e


# Verify install
dotnet --info


# Where did java get installed?
Get-ChildItem "C:\Program Files\Microsoft" -Directory | Where-Object Name -match '^jdk-17' | Select-Object FullName

# ---- adjust this if your folder name differs ----
$JdkHome = "C:\Program Files\Microsoft\jdk-17.0.16.8-hotspot"

# Make it work *right now* in this PowerShell session:
$env:JAVA_HOME = $JdkHome
$env:Path += ";$JdkHome\bin"

# Also set for the whole machine (persists for new shells):
setx JAVA_HOME "$JdkHome" /M
setx PATH "$env:Path" /M


# (this seemed to fail for me, was that a problem?)
java -version

##-----
### Is this necessary? I think it is not.
## Check what nuget thinks we have:
#dotnet nuget list source --format detailed
##Add sources
## NuGet.org (primary)
#dotnet nuget add source https://api.nuget.org/v3/index.json -n nuget.org
## Microsoft’s public feed used by some workloads (safe to add)
#dotnet nuget add source https://pkgs.dev.azure.com/dnceng/public/_packaging/dotnet-public/nuget/v3/index.json -n dotnet-public
## Confirm
#dotnet nuget list source
##-----


# Install the android SDK
Write-Host "Installing Android SDK command-line tools (manual fallback)"
$SdkRoot = "$env:LOCALAPPDATA\Android\Sdk"
New-Item -ItemType Directory -Path $SdkRoot -Force | Out-Null

# Download Google’s command-line tools zip (≈10 MB)
$zip = "$env:TEMP\commandlinetools.zip"
Write-Host $zip
$ProgressPreference = 'SilentlyContinue'
Invoke-WebRequest -Uri "https://dl.google.com/android/repository/commandlinetools-win-11076708_latest.zip" -OutFile $zip

Expand-Archive -Path $zip -DestinationPath "$SdkRoot\cmdline-tools\temp" -Force
#Remove-Item "$SdkRoot\cmdline-tools\latest" -Recurse -Force -ErrorAction SilentlyContinue
Move-Item "$SdkRoot\cmdline-tools\temp" "$SdkRoot\cmdline-tools\latest" -Force
Remove-Item $zip

# Fix issue with putting the file in the wrong place
$SdkRoot = "$env:LOCALAPPDATA\Android\Sdk"
$nested = Join-Path $SdkRoot "cmdline-tools\latest\cmdline-tools"
$target = Join-Path $SdkRoot "cmdline-tools\latest"

if (Test-Path $nested) {
    Write-Host "Flattening nested cmdline-tools directory..." -ForegroundColor Cyan
    Get-ChildItem -Path $nested -Force | Move-Item -Destination $target -Force
    Remove-Item $nested -Recurse -Force
    Write-Host "✅ Fixed: cmdline-tools now properly at $target" -ForegroundColor Green
} else {
    Write-Host "No nested cmdline-tools folder detected — nothing to fix." -ForegroundColor Yellow
}


# Set environment vars for this session
$env:ANDROID_HOME = $SdkRoot
$env:ANDROID_SDK_ROOT = $SdkRoot
[Environment]::SetEnvironmentVariable("ANDROID_HOME", $SdkRoot, "User")
[Environment]::SetEnvironmentVariable("ANDROID_SDK_ROOT", $SdkRoot, "User")

$SdkRoot = "$env:LOCALAPPDATA\Android\Sdk"
$yes = ("y`n" * 50)
$yes | & "$SdkRoot\cmdline-tools\latest\bin\sdkmanager.bat" --sdk_root="$SdkRoot" --licenses


# Install platform tools and latest API
& "$SdkRoot\cmdline-tools\latest\bin\sdkmanager.bat" --sdk_root=$SdkRoot `
    "platform-tools" `
    "platforms;android-34" `
    "build-tools;34.0.0" `
    "cmdline-tools;latest"


# Add MUAI worloads
dotnet workload install maui-android
dotnet workload list  # sanity check; look for maui-android 9.x

# maybe installs the ios dependencies that the project wants?
dotnet workload restore


<#

VM_NAME=WinDev2407Eval
VBoxManage startvm "$VM_NAME"

ssh -p 2222 user@localhost

cd $HOME/code/shitspotter/tpl/poopdetector

git pull 
git checkout flexible_model_config

#>



# Move the models to the appropriate place to bake them in
ls ..\poop_models
ls .\PoopDetector\Resources\Raw\
cp ..\poop_models\shitspotter-custom-v5-epoch_115.onnx .\PoopDetector\Resources\Raw\shitspotter-custom-v5-epoch_115.onnx
cp ..\poop_models\shitspotter_custom_v2_epoch126.onnx .\PoopDetector\Resources\Raw\shitspotter_custom_v2_epoch126.onnx

# Place to add new models is:
# ~/code/shitspotter/tpl/poopdetector/PoopDetector/AI/WithSession/VisionModelManager.cs
# ~/code/shitspotter/tpl/poopdetector/PoopDetector/PoopDetector.csproj
# ~/code/shitspotter/tpl/poopdetector/PoopDetector/appsettings.json

# Build and install dependencies
$AndroidSdk='C:\work\android-sdk'
New-Item -ItemType Directory -Force -Path $AndroidSdk | Out-Null
dotnet build -t:InstallAndroidDependencies -f net9.0-android `
  -p:AndroidSdkDirectory=$AndroidSdk `
  -p:JavaSdkDirectory=$env:JAVA_HOME `
  -p:AcceptAndroidSdkLicenses=True `


# This seemed to work.
$AndroidSdk='C:\work\android-sdk'
$JdkHome = "C:\work\jdk"
dotnet publish -f net9.0-android -c Release `
  -p:AndroidPackageFormat=apk `
  -p:RunAOTCompilation=false `
  -p:AndroidSupportedAbis=arm64-v8a `
  -p:AndroidSdkDirectory=$AndroidSdk `
  -p:JavaSdkDirectory=$JdkHome

# Build apk package should be here
ls .\PoopDetector\bin\Release\net9.0-android\com.poop.detector-Signed.apk



############



### ON linux

# Copy the built apk from the windows VM to the linux host

# Note: the windows VM needs to be closed to do this apparently...
VM_NAME=WinDev2407Eval
VBoxManage controlvm "$VM_NAME" pause
VM_NAME=WinDev2407Eval
VBoxManage controlvm "$VM_NAME" resume

VBoxManage controlvm "$VM_NAME" poweroff

# To resume later
#VBoxManage startvm "$VM_NAME" --type headless


# Run the linux emulator
export ANDROID_SDK_ROOT="$HOME/Android/Sdk"
echo 'export PATH="$ANDROID_SDK_ROOT/platform-tools:$PATH"' >> ~/.bashrc
export PATH="$ANDROID_SDK_ROOT/platform-tools:$PATH"
#"$ANDROID_SDK_ROOT"/emulator/emulator -avd pixel_34 -no-boot-anim -netdelay none -netspeed full 
$ANDROID_SDK_ROOT/emulator/emulator -avd pixel_34 \
  -no-accel \
  -gpu swiftshader_indirect \
  -no-snapshot \
  -no-boot-anim \
  -skin 480x800 -memory 2048



adb devices

# Install the apk to the 
adb install -r -d -t com.poop.detector-Signed.apk 



# 

#ssh -p 2222 user@localhost
scp -P 2222 user@localhost:code/shitspotter/tpl/poopdetector/PoopDetector/bin/Release/net9.0-android/com.poop.detector-Signed.apk com.poop.detector-Signed.apk

adb devices

adb install -r -d -t com.poop.detector-Signed.apk 

#rsync -avRP gadget:code/poopdetector/poopdetector/PoopDetector/bin/Release/net9.0-android/./com.poop.detector.apk .
#rsync -avRP gadget:code/poopdetector/poopdetector/PoopDetector/bin/Release/net9.0-android/./com.poop.detector-Signed.apk .
 
adb devices
adb install -r -d -t com.poop.detector.apk .
adb install -r -d -t com.poop.detector-Signed.apk 
adb install -r com.poop.detector-Signed.apk 
