# 6\. Asset Change Test

## Learning Objectives

This exercise will teach you a popular pattern in Game Tests to verify if Assets change over time.

## Exercise

As you noticed inside LostCrypt, when you pick up the Wand, your character equips armor.  
Write a test that checks that after Sara picks up the wand, armor is equipped.  

1.  Create a `WandTests.cs` class and implement `MainScene\_CharacterReachesWandAndEquipsArmor` test.
2.  Try to observe how `Sara Variant` or more specifically `puppet_sara` GameObject changes the moment you pick up the wand.

## Hints

*   You can reuse code from [Reach Wand Test](./reach-wand-test.md) for the logic of the character picking up the wand. Or you can try to trigger this action programmatically.
*   Remember that if some Unity internal APIs are not accessible for your test you might need to add a new reference inside the `PlayModeTests` assembly definition.

## Solution

PlayModeTests.asmdef  
```
{
    "name": "PlayModeTests",
    "rootNamespace": "",
    "references": [
        "Unity.InputSystem",
        "Unity.InputSystem.TestFramework",
        "TestInputControl",
        "UnityEngine.TestRunner",
        "Unity.2D.Animation.Runtime"
    ],
    "includePlatforms": [],
    "excludePlatforms": [],
    "allowUnsafeCode": false,
    "overrideReferences": true,
    "precompiledReferences": [
        "nunit.framework.dll"
    ],
    "autoReferenced": false,
    "defineConstraints": [
        "UNITY_INCLUDE_TESTS"
    ],
    "versionDefines": [],
    "noEngineReferences": false
}
``` 

WandTests.cs  
```
using System.Collections;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;
using UnityEngine.SceneManagement;
using UnityEngine.Experimental.U2D.Animation;

public class WandTests
{
    private Transform _characterTransform;
    private float _testTimeout = 25.0f;
    private float _wandLocation = 21.080f;

    [UnityTest]
    public IEnumerator MainScene_CharacterReachesWandAndEquipsArmor()
    {
        SceneManager.LoadScene("Assets/Scenes/Main.unity", LoadSceneMode.Single);
        
        // Skip first frame so Sara have a chance to appear on the screen
        yield return null;
        var puppet = GameObject.Find("puppet_sara");
        var spriteLibrary = puppet.GetComponent<SpriteLibrary>();
        
        Assert.AreEqual(spriteLibrary.spriteLibraryAsset.name, "Sara");

        var elapsedTime = 0.0f;
        yield return GoRight();
        while (GetCurrentCharacterPosition() <= _wandLocation)
        {
            yield return null;
            elapsedTime += Time.deltaTime;
            if (elapsedTime > _testTimeout)
            {
                Assert.Fail($"Character did not reach location position in {_testTimeout} seconds.");
            }
        }

        // Wait for Wand pickup animation to be over.
        yield return new WaitForSeconds(12);

        Assert.AreEqual(spriteLibrary.spriteLibraryAsset.name, "Sara_var01");
    }

    private float GetCurrentCharacterPosition()
    {
        // Get Main character's Transform which is used to manipulate position.
        if (_characterTransform == null)
        {
            _characterTransform = GameObject.Find("Sara Variant").transform;
        }

        return _characterTransform.position.x;
    }

    private IEnumerator GoRight()
    {
        TestInputControl.MoveLeft = false;
        yield return null;
        TestInputControl.MoveRight = true;
    } 
}
```
