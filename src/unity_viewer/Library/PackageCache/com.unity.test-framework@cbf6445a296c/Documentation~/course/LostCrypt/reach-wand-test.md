# 4\. Reach Wand Test

## Learning Objectives

Perform Assertions on your character position and behavior.

## Exercise

1.  Go back to the previous `MovementTest.cs` file.
2.  Write a `MainScene\_CharacterReachesWand` test that makes your character move right, and checks if it reaches the wand location.

## Hints

*   Look for _Altar_ and _Sara Variant_ game objects in your scene. You are interested in measuring the X position of your Transform objects.
*   Wand location X position is equal float of **21.080**. Main Character X position is dynamic and it changes whenever it moves.
*   Consider setting a timeout that makes the test fail if the Wand is not reached.

## Solution

```
using System.Collections;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;
using UnityEngine.SceneManagement;

public class MovementTest
{
    private Transform _characterTransform;
    private float _testTimeout = 25.0f;
    private float _wandLocation = 21.080f;

    [UnityTest]
    public IEnumerator MainScene_CharacterReachesWand()
    {
      SceneManager.LoadScene("Assets/Scenes/Main.unity", LoadSceneMode.Single);
      yield return waitForSceneLoad();

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

    private IEnumerator waitForSceneLoad()
    {
        while (SceneManager.GetActiveScene().buildIndex > 0)
        {
            yield return null;
        }
    }
}
```
