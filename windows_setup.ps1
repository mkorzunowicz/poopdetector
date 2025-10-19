# In VSCode I also installed the recommended C# dev kit, not sure if that matters.

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

#-----
## Is this necessary?
# Check what nuget thinks we have:
dotnet nuget list source --format detailed
#Add sources
# NuGet.org (primary)
dotnet nuget add source https://api.nuget.org/v3/index.json -n nuget.org
# Microsoft’s public feed used by some workloads (safe to add)
dotnet nuget add source https://pkgs.dev.azure.com/dnceng/public/_packaging/dotnet-public/nuget/v3/index.json -n dotnet-public
# Confirm
dotnet nuget list source
#-----



# Add MUAI worloads

dotnet workload install maui-android
dotnet workload list  # sanity check; look for maui-android 9.x


# maybe installs the ios dependencies that the project wants?
dotnet workload restore

# One-shot: install Android SDK/NDK & accept licenses (CLI)
# Choose folders without spaces; adjust as you like
#New-Item -ItemType Directory -Force -Path $AndroidSdk | Out-Null
#dotnet build -t:InstallAndroidDependencies -f net9.0-android `
#  -p:AndroidSdkDirectory=$AndroidSdk `
#  -p:JavaSdkDirectory=$env:JAVA_HOME `

# This build command did not work 
$AndroidSdk='C:\work\android-sdk'
$JdkHome = "C:\work\jdk"
dotnet publish -t:InstallAndroidDependencies `
  -p:AndroidSdkDirectory=$AndroidSdk `
  -p:JavaSdkDirectory=$JdkHome `
  -p:AcceptAndroidSdkLicenses=True `
  -f net9.0-android -c Release -p:AndroidPackageFormat=apk


# try again
dotnet clean
dotnet publish -f net9.0-android -c Release `
  -p:AndroidPackageFormat=apk `
  -p:RunAOTCompilation=false `
  -p:AndroidSupportedAbis=arm64-v8a


# This seemed to work.
$AndroidSdk='C:\work\android-sdk'
$JdkHome = "C:\work\jdk"
dotnet publish -f net9.0-android -c Release `
  -p:AndroidPackageFormat=apk `
  -p:RunAOTCompilation=false `
  -p:AndroidSupportedAbis=arm64-v8a `
  -p:AndroidSdkDirectory=$AndroidSdk `
  -p:JavaSdkDirectory=$JdkHome


ls .\PoopDetector\bin\Release\net9.0-android\com.poop.detector.apk

### Install android tools

winget install --id Google.AndroidSDK.PlatformTools -e


### ON linux

rsync -avRP gadget:code/poopdetector/poopdetector/PoopDetector/bin/Release/net9.0-android/./com.poop.detector.apk .
rsync -avRP gadget:code/poopdetector/poopdetector/PoopDetector/bin/Release/net9.0-android/./com.poop.detector-Signed.apk .
 
adb devices
adb install -r -d -t com.poop.detector.apk .
adb install -r -d -t com.poop.detector-Signed.apk 
adb install -r com.poop.detector-Signed.apk 
