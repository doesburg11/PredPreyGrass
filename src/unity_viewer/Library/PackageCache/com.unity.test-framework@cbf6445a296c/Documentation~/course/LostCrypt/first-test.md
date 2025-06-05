# 2\. Running a test in a LostCrypt

## Learning Objectives

Set up a simple Play Mode test for LostCrypt.

## Exercise

1.  Go to the `Assets/Scripts` directory, and spend some time exploring the scripts necessary for LostCrypt to work properly.
2.  Create a new directory `Assets/Tests`.
3.  In the Test Runner window click **Create PlayModeTest Assembly Folder** and name a new folder `PlayModeTests`. You should end up with `Assets/Tests/PlayModeTests`.
4.  Open the newly created folder and click **Create Test Script in current folder** in the Test Runner window.
5.  Name the file `SceneSetupTests.cs`.
6.  Write your first test that asserts that after loading the Main scene the current time is day.

## Hints

*   In order to load scenes, please refer to [UnityEngine.SceneManagement](https://docs.unity3d.com/ScriptReference/SceneManagement.SceneManager.html) documentation.
*   Inside `Scenes/Main.unity` [look for GameObject](https://docs.unity3d.com/ScriptReference/GameObject.Find.html) **FX - Day**.

## Solution
  
SceneSetupTests.cs 

```
using System.Collections;
using System.Collections.Generic;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;
using UnityEngine.SceneManagement;

public class SceneSetupTests
{
    [UnityTest]
    public IEnumerator MainScene_LoadsCorrectlyAndItsDaytime()
    {
        SceneManager.LoadScene("Assets/Scenes/Main.unity", LoadSceneMode.Single);
        yield return null;

        var fxDay = GameObject.Find("FX - Day");

        Assert.IsTrue(fxDay != null, "should find the 'FX - Day' object in the scene");
    }
}
```
