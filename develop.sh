__doc__='
References:
    https://learn.microsoft.com/en-us/dotnet/core/install/linux-ubuntu-install?tabs=dotnet9&pivots=os-linux-ubuntu-2404
'

curl -fsSL https://dot.net/v1/dotnet-install.sh -o dotnet-install.sh
chmod +x dotnet-install.sh
./dotnet-install.sh --channel 9.0 --install-dir "$HOME/.dotnet"

echo 'export PATH="$HOME/.dotnet:$PATH"' >> ~/.bashrc
export PATH="$HOME/.dotnet:$PATH"


dotnet --info


sudo apt install openjdk-17-jdk
java -version


dotnet workload list


curl -fsSL https://dot.net/v1/dotnet-install.sh -o dotnet-install.sh
chmod +x dotnet-install.sh

# install the latest .NET 9 SDK to ~/.dotnet
./dotnet-install.sh --channel 9.0 --install-dir "$HOME/.dotnet"

which dotnet   # should now be ~/.dotnet/dotnet
dotnet --info
dotnet workload search | grep android   # should now show "android"

dotnet workload install android
dotnet workload install maui-android
dotnet workload list


# 1) Choose where your SDK will live
export ANDROID_SDK_ROOT="$HOME/Android/Sdk"

# 2) Create folders
mkdir -p "$ANDROID_SDK_ROOT/cmdline-tools"

# 3) Download the Command-line Tools (Linux) zip from Google
# (get the link from the official page below)
# Example (version number may change):
cd /tmp
curl -LO https://dl.google.com/android/repository/commandlinetools-linux-11076708_latest.zip

# 4) Unzip into the expected 'latest' layout
unzip -q commandlinetools-linux-*_latest.zip -d "$ANDROID_SDK_ROOT/cmdline-tools"
# The zip extracts a 'cmdline-tools' folder; move it to 'latest'
mv "$ANDROID_SDK_ROOT/cmdline-tools/cmdline-tools" "$ANDROID_SDK_ROOT/cmdline-tools/latest"

# 5) Accept licenses (now sdkmanager exists)
yes | "$ANDROID_SDK_ROOT/cmdline-tools/latest/bin/sdkmanager" --licenses

# 6) Install essentials
"$ANDROID_SDK_ROOT/cmdline-tools/latest/bin/sdkmanager" --sdk_root="$ANDROID_SDK_ROOT" \
  "platform-tools" "platforms;android-35" "build-tools;35.0.0"




# Make sure Android Studio is installed & ANDROID_SDK_ROOT is set
export ANDROID_SDK_ROOT="$HOME/Android/Sdk"

# Accept licenses
yes | "$ANDROID_SDK_ROOT/cmdline-tools/latest/bin/sdkmanager" --licenses

# Check that your emulator/device is visible
adb devices

export ANDROID_SDK_ROOT="$HOME/Android/Sdk"
echo 'export PATH="$ANDROID_SDK_ROOT/platform-tools:$PATH"' >> ~/.bashrc
export PATH="$ANDROID_SDK_ROOT/platform-tools:$PATH"

# Build & run the app (in the repo root)
cd ~/code/shitspotter/tpl/poopdetector
dotnet build -t:Run -f net9.0-android

cd ~/code/shitspotter/tpl/poopdetector

dotnet restore PoopDetector.sln -p:TargetFramework=net9.0-android
dotnet build   PoopDetector.sln -p:TargetFramework=net9.0-android

# for Run, specify the project explicitly (solutions don’t always know the startup proj):
#dotnet build PoopDetector/PoopDetector.csproj -t:Run -f net9.0-android




$ANDROID_SDK_ROOT/cmdline-tools/latest/bin/sdkmanager --install \
  "emulator" "platform-tools" "platforms;android-34" "system-images;android-34;google_apis;x86_64"

# List installed system images
$ANDROID_SDK_ROOT/cmdline-tools/latest/bin/avdmanager list system-images

# Create an AVD called "pixel_34"
$ANDROID_SDK_ROOT/cmdline-tools/latest/bin/avdmanager create avd \
  -n pixel_34 -k "system-images;android-34;google_apis;x86_64" \
  -d pixel_6

$ANDROID_SDK_ROOT/emulator/emulator -avd pixel_34 -netdelay none -netspeed full &

adb devices

dotnet run --project PoopDetector/PoopDetector.csproj -f net9.0-android -p:AdbTarget=emulator-5554



####

dotnet clean PoopDetector/PoopDetector.csproj
rm -rf PoopDetector/bin PoopDetector/obj

# restore & build ONLY Android
dotnet restore PoopDetector/PoopDetector.csproj -f net9.0-android --force
dotnet build   PoopDetector/PoopDetector.csproj -f net9.0-android

# run on the emulator/device
dotnet run --project PoopDetector/PoopDetector.csproj -f net9.0-android


# ---

cd ~/code/shitspotter/tpl/poopdetector

# Clean Android only
dotnet clean PoopDetector/PoopDetector.csproj -p:TargetFrameworks=net9.0-android

# Restore Android only  (note: restore uses -p:TargetFramework=..., not -f)
dotnet restore PoopDetector/PoopDetector.csproj -p:TargetFramework=net9.0-android --force

# Build & run on the running emulator/device (here -f is correct)
dotnet build PoopDetector/PoopDetector.csproj -t:Run -f net9.0-android
# or:
dotnet run --project PoopDetector/PoopDetector.csproj -f net9.0-android


##

$ANDROID_SDK_ROOT/emulator/emulator -avd pixel_34 -netdelay none -netspeed full &

adb install -r /home/joncrall/Downloads/app.apk

adb install -r -d -t /home/joncrall/Downloads/app.apk


$ANDROID_SDK_ROOT/emulator/emulator -avd pixel_34 -camera-back webcam0

$ANDROID_SDK_ROOT/emulator/emulator -avd pixel_34 -netdelay none -netspeed full -camera-back webcam0
adb install -r -d -t /home/joncrall/Downloads/app.apk

