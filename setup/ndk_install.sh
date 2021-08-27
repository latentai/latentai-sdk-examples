#!/bin/bash

cd /opt  
wget https://dl.google.com/android/repository/android-ndk-r21-linux-x86_64.zip && unzip android-ndk-r21-linux-x86_64.zip && rm -rf android-ndk-r21-linux-x86_64.zip
export ANDROID_NDK_HOME=/opt/android-ndk-r21 
$ANDROID_NDK_HOME/build/tools/make-standalone-toolchain.sh --platform=android-24 --use-llvm --arch=arm64 --install-dir=/opt/android-toolchain-arm64
export LEIP_NDK_CC=/opt/android-toolchain-arm64/bin/aarch64-linux-android-g++
