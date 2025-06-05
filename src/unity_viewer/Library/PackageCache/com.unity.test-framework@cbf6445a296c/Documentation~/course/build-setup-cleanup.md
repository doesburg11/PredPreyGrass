# 12\. Setup and cleanup at build time

## Learning objectives

This section will introduce you to the hooks in the test framework for before and after the player build.

## Intro and motivation

Sometimes it's necessary to change settings or prepare assets before a build for Play Mode tests. Similarly, it might be relevant to clean up things after the build. For this the test framework has two hookup points called [PrebuildSetup and PostBuildCleanup](https://docs.unity3d.com/Packages/com.unity.test-framework@1.1/manual/reference-setup-and-cleanup.html).  
  
In the Editor, the `PrebuildSetup` is invoked before the build and test run and the `PostBuildCleanup` is invoked after the tests are completely done. This happens for both Edit Mode and Play Mode tests. When running Play Mode tests on a device, the Cleanup is already run right after the build is done, as the tests are happening in parallel on the device.  
  
The simplest way of ensuring a test has a `PrebuildSetup` and `PostBuildCleanup` is by implementing `IPrebuildSetup` and `IPostBuildCleanup` respectively in your test class.  
  
Often the setup and cleanup will be interacting with code in the `UnityEditor` assemblies. These are not available when running on a device, but we want our built-in setup and cleanup code to stay in the test class. For this, it's recommended to wrap the Editor-related code lines in `#if UNITY_EDITOR` defines. For example:  

```
public class MyTestClass : IPrebuildSetup
{
 [Test]
 public void MyTest()
 {
 
 }

 public void Setup()
 {
 #if UNITY_EDITOR
   UnityEditor.EditorSettings.serializationMode = SerializationMode.ForceText;
 #endif
 }
}
```
  
> Note: If the Editor code is not wrapped, then you won't see any compilation error when running in the Editor, but you will see the compilation error once you try to run the test in a player.

## Exercise

The sample `12_BuildSetupCleanup` contains a Play Mode test for verifying the content of a scene. It is essentially the Play Mode version of the test from the previous exercise.  
  
The test fails because the scene can't be found. It could be solved by adding the scene to the build settings, but it's not good practise to add a test-related scene to the build settings, as it could get included when building for non-testing purposes.  
  
Therefore the task is to create a `PrebuildSetup` that adds the scene to `EditorBuildSettings` and a `PostBuildCleanup` that removes it again.  
  
Test the solution by running the test both in the Editor and in a standalone player. You will need to use `#if UNITY_EDITOR` to make the code compile for the player.  

## Hints

*   The `IPrebuildSetup` interface requires a `Setup` method, so be careful that there are no `[SetUp]` methods already called that.

## Solution

A full solution is available in the [sample](./welcome.md#import-samples) `12_BuildSetupCleanup_Solution`.  
  
The full test solution can be done like this:

```
using System.Collections;
using System.Linq;
using NUnit.Framework;
#if UNITY_EDITOR
using UnityEditor;
#endif
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.TestTools;

namespace Tests
{
 public class SceneTests : IPrebuildSetup, IPostBuildCleanup
 {
  private string originalScene;
  private const string k_SceneName = "Assets/MyGameScene.unity";

  public void Setup()
  {
#if UNITY_EDITOR
   if (EditorBuildSettings.scenes.Any(scene => scene.path == k_SceneName))
   {
    return;
   }
   
   var includedScenes = EditorBuildSettings.scenes.ToList();
   includedScenes.Add(new EditorBuildSettingsScene(k_SceneName, true));
   EditorBuildSettings.scenes = includedScenes.ToArray();
#endif
  }

  [UnitySetUp]
  public IEnumerator SetupBeforeTest()
  {
   originalScene = SceneManager.GetActiveScene().path;
   SceneManager.LoadScene(k_SceneName);
   yield return null; // Skip a frame
  }

  [Test]
  public void VerifyScene()
  {
   var gameObject = GameObject.Find("GameObjectToTestFor");

   Assert.That(gameObject, Is.Not.Null, $"GameObjectToTestFor not found in {SceneManager.GetActiveScene().path}.");
  }

  [TearDown]
  public void TeardownAfterTest()
  {
   SceneManager.LoadScene(originalScene);
  }

  public void Cleanup()
  {
#if UNITY_EDITOR
   EditorBuildSettings.scenes = EditorBuildSettings.scenes.Where(scene => scene.path != k_SceneName).ToArray();
#endif
  }
 }
}
```

Note that `#if UNITY_EDITOR` is also used among the using statements, to allow for a using reference to `UnityEditor`.

## Further reading and resources

[UTF documentation for PrebuildSetup and PostBuildCleanup](https://docs.unity3d.com/Packages/com.unity.test-framework@1.1/manual/reference-setup-and-cleanup.html).
