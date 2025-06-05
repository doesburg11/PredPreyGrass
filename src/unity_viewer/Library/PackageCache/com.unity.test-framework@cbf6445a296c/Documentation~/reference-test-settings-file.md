# Test settings file

You can define options for a test run in a `TestSettings.json` file. You can specify the location of this file using the [testSettingsFile](./reference-command-line.md#testsettingsfile) command line argument.

Here's an example of a `TestSettings.json` file:

```json
{
  "scriptingBackend":"WinRTDotNET",
  "Architecture":null,
  "apiProfile":0,
  "featureFlags": { "requiresSplashScreen": true }
}
```

## Supported options

You can define the following options in a test settings file:

### apiProfile

The .Net compatibility level, refer to [ApiCompatabilityLevel](https://docs.unity3d.com/ScriptReference/ApiCompatibilityLevel.html). Set to one of the following values:  

- 1 - .Net 2.0 
- 2 - .Net 2.0 Subset 
- 3 - .Net 4.6 
- 5 - .Net micro profile (used by Mono scripting backend if **Stripping Level** is set to **Use micro mscorlib**) 
- 6 - .Net Standard 2.0 

### appleEnableAutomaticSigning

Sets option for automatic signing of Apple devices, refer to [PlayerSettings.iOS.appleEnableAutomaticSigning](https://docs.unity3d.com/ScriptReference/PlayerSettings.iOS-appleEnableAutomaticSigning.html).

### appleDeveloperTeamID 

Sets the team ID for the apple developer account, refer to [PlayerSettings.iOS.appleDeveloperTeamID](https://docs.unity3d.com/ScriptReference/PlayerSettings.iOS-appleDeveloperTeamID.html).

### architecture

Target architecture for Android, refer to [AndroidArchitecture](https://docs.unity3d.com/ScriptReference/AndroidArchitecture.html). Set to one of the following values: 

* None = 0
* ARMv7 = 1
* ARM64 = 2
* X86 = 4
* All = 4294967295

### iOSManualProvisioningProfileType

Refer to [PlayerSettings.iOS.iOSManualProvisioningProfileType](https://docs.unity3d.com/ScriptReference/PlayerSettings.iOS-iOSManualProvisioningProfileType.html). Set to one of the following values: 

* 0 - Automatic 
* 1 - Development 
* 2 - Distribution [iOSManualProvisioningProfileID](https://docs.unity3d.com/ScriptReference/PlayerSettings.iOS-iOSManualProvisioningProfileID.html)

### iOSTargetSDK

Target SDK for iOS. Set to one of the following values, which should be given as a string literal enclosed in quotes:

* DeviceSDK
* SimulatorSDK

### tvOSManualProvisioningProfileType

Refer to [PlayerSettings.iOS.tvOSManualProvisioningProfileType](https://docs.unity3d.com/ScriptReference/PlayerSettings.iOS-tvOSManualProvisioningProfileType.html). Set to one of the following values: 

* 0 - Automatic 
* 1 - Development 
* 2 - Distribution [tvOSManualProvisioningProfileID](https://docs.unity3d.com/ScriptReference/PlayerSettings.iOS-tvOSManualProvisioningProfileID.html)

### tvOSTargetSDK

Target SDK for tvOS. Set to one of the following values, which should be given as a string literal enclosed in quotes:

* DeviceSDK
* SimulatorSDK

### scriptingBackend

 Set to one of the following values, which should be given as a string literal enclosed in quotes:

- Mono2x
- IL2CPP
- WinRTDotNET

### playerGraphicsAPI

 Set graphics API that will be used during test execution in the player. Value can be any [GraphicsDeviceType](https://docs.unity3d.com/ScriptReference/Rendering.GraphicsDeviceType.html) as a string literal enclosed in quotes. Value will only be set if it is supported on the target platform.

### webGLClientBrowserType

A browser to be used when running test using WebGL platform. Accepted browser types:

- Safari
- Firefox
- Chrome
- Chromium

### webGLClientBrowserPath

 An absolute path to the browser's location on your device. If not defined, path from UNITY_AUTOMATION_DEFAULT_BROWSER_PATH enviromental variable will be used.

### androidBuildAppBundle

A boolean setting that allows to build an Android App Bundle (AAB) instead of APK for tests.

### featureFlags

Map of strings and boolean values which can switch Unity Test Framework features on or off. The currently supported features are:

* fileCleanUpCheck
Throws an error message (instead of warning) if tests generate files which are not cleaned up. False (off) by default.

* requiresSplashScreen
By default UTR disables the Made with Unity splash screen to speed up building the player and running tests. Set this flag to `true` to override the default and always require a splash screen to be built.


