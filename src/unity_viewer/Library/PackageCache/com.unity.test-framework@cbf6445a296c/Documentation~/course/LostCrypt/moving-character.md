# 3\. Moving character

## Learning objectives

How to use the Unity InputSystem package to have a generic way of moving your character programmatically in tests.

## Exercise

Please make sure [InputSystem](https://docs.unity3d.com/Packages/com.unity.inputsystem@1.0/manual/QuickStartGuide.html) is installed in your Unity project. You can verify that by checking the Package Manager.

1.  Create a new class called `MovementTest.cs` under `Assets/Tests/PlayModeTests`.
2.  Attach the reference to `Unity.InputSystem` and `Unity.InputSystem.TestFramework` in your `PlayModeTests` assembly definition.
3.  Create a new `InputControl` directory under Tests: `Assets/Tests/InputControl`.
4.  Inside `InputControl` directory, create a new assembly definition: `TestInputControl.asmdef`.
5.  Create a new class `TestInputControl.cs` where you implement following properties:
```
public static bool MoveLeft { get; set; }
public static bool MoveRight { get; set; }
public static bool Jump { get; set; }
```
6.  Go back to your assembly definition `PlayModeTests` and attach the reference to newly created: `TestInputControl`.
7.  Finally, we need to use our `TestInputControl` in actual LostCrypt code. Currently Unity's `InputSystem` does not support an easier way of programmatically doing mocks, please see this git diff to know what to change inside `CharacterController2D`:

```
diff --git a/Assets/Scripts/CharacterController2D.cs b/Assets/Scripts/CharacterController2D.cs
index f8a10cf2..e0a62878 100644
--- a/Assets/Scripts/CharacterController2D.cs
+++ b/Assets/Scripts/CharacterController2D.cs
@@ -81,15 +81,15 @@ public class CharacterController2D : MonoBehaviour
         // Horizontal movement
         float moveHorizontal = 0.0f;
 
-        if (keyboard.leftArrowKey.isPressed || keyboard.aKey.isPressed)
+        if (keyboard.leftArrowKey.isPressed || keyboard.aKey.isPressed || TestInputControl.MoveLeft)
             moveHorizontal = -1.0f;
-        else if (keyboard.rightArrowKey.isPressed || keyboard.dKey.isPressed)
+        else if (keyboard.rightArrowKey.isPressed || keyboard.dKey.isPressed || TestInputControl.MoveRight)
             moveHorizontal = 1.0f;
 
         movementInput = new Vector2(moveHorizontal, 0);
 
         // Jumping input
-        if (!isJumping && keyboard.spaceKey.wasPressedThisFrame)
+        if (!isJumping && (keyboard.spaceKey.wasPressedThisFrame || TestInputControl.Jump))
             jumpInput = true;
     } 
```

Now you are ready! Go back to `MovementTest.cs` and write a test that does not do any assertions (just yet), but only moves the Sara character and makes it occasionally jump.  

## Hints

*   You might want to use `WaitForSeconds` in your test, to deliberately make it run longer and see actual animation happening on your screen.
*   In case of compilation issues, please make sure you follow the right folder structure:
  
```
Tests
    InputControl
        TestInputControl.asmdef
        TestInputControl.cs
    PlayModeTests
        MovementTest.cs
        PlayModeTest.asmdef
```

## Solution
  
PlayModeTests.asmdef  
```
{
    "name": "PlayModeTests",
    "references": [
      "Unity.InputSystem",
      "Unity.InputSystem.TestFramework",
      "TestInputControl"
    ],
    "optionalUnityReferences": [
      "TestAssemblies"
    ]
}
``` 
MovementTest.cs  
```
using System.Collections;
using UnityEngine;
using UnityEngine.TestTools;
using UnityEngine.SceneManagement;

public class MovementTest
{
    [UnityTest]
    public IEnumerator MainScene_CharacterIsAbleToJump()
    {
      SceneManager.LoadScene("Assets/Scenes/Main.unity", LoadSceneMode.Single);
      yield return waitForSceneLoad();
      yield return GoRight();
      yield return new WaitForSeconds(2);
      yield return Jump();
      yield return new WaitForSeconds(3);
      yield return GoLeft();
      yield return Jump();
      yield return new WaitForSeconds(2);
    }

    private IEnumerator Jump()
    {
      TestInputControl.Jump = true;
      yield return null;
      TestInputControl.Jump = false;
    }

    private IEnumerator GoRight()
    {
      TestInputControl.MoveLeft = false;
      yield return null;
      TestInputControl.MoveRight = true;
    }

    private IEnumerator GoLeft()
    {
      TestInputControl.MoveRight = false;
      yield return null;
      TestInputControl.MoveLeft = true;
    }

    private IEnumerator waitForSceneLoad()
    {
        while (SceneManager.GetActiveScene().buildIndex > 0)
        {
            yield return null;
        }
    }
}
``` 
