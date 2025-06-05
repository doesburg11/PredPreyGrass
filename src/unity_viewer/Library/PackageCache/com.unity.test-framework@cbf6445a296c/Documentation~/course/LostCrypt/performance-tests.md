# 8\. Performance Tests

## Learning Objectives

One final thing we'll explore is a package that extends Unity Test Framework with Performance Tests.

## Exercise

The Performance Testing package can be used to measure performance in our game. This is a great tool if we want to track various regressions/progressions that happen over time in our project. In this example, you'll learn how to create a test that measures game average frames.

1.  LostCrypt does not include the Performance Testing package installed by default. Install it by following [these instructions](https://docs.unity3d.com/Packages/com.unity.test-framework.performance@2.8/manual/index.html).
2.  Add the package as a dependency to the [project manifest](https://docs.unity3d.com/Manual/upm-manifestPrj.html).
3.  When the package is installed, add a reference to `Unity.PerformanceTesting` in your **PlayModeTests** assembly definition to access the performance testing APIs.
4.  Create a new C# class under **Assets/Tests/PlayModeTests** called **PerformanceTests.cs**.

You're now ready to complete your objective. In `PerformanceTests.cs` create a new function called `MainScene_MeasureAverageFrames()`. In this function move your character to the wand position and wait until the wand pickup effect is over. During all that time, measure the frames.  

## Bonus

*   Try to measure the average FPS in LostCrypt. You might need to use `Time.deltaTime` from UnityEngine API and `Measure.Custom` from the Performance Testing package API.

## Hints

*   The first handful of frames after loading Scene are usually unstable, let's utilize the `Measure.Frames().Scope()` API to measure them into a separate scope.
*   After your test finishes, performance results can be viewed under **Window > Analysis > Performance Test Report** or you can even hook into results using [Callback API](https://docs.unity3d.com/Packages/com.unity.test-framework@1.1/manual/extension-get-test-results.html).

## Solution
  
PerformanceTests.cs 
```
using System.Collections;
using Unity.PerformanceTesting;
using UnityEngine;
using UnityEngine.TestTools;
using UnityEngine.SceneManagement;

public class PerformanceTests
{
    private Transform _characterTransform;
    private float _wandLocation = 21.080f;
        
    [UnityTest, Performance]
    public IEnumerator MainScene_MeasureAverageFrames()
    {
        SceneManager.LoadScene("Assets/Scenes/Main.unity", LoadSceneMode.Single);
        using (Measure.Frames().Scope("Frames.MainSceneOnLoad.Unstable"))
        {
            for (var i = 0; i < 25; i++)
            {
                yield return null;
            }
        }

        using (Measure.Frames().Scope("Frames.MainSceneGameplay"))
        {
            yield return GoRight();
            while (GetCurrentCharacterPosition() <= _wandLocation)
            {
                yield return null;
            }

            StopMoving();
            yield return new WaitForSeconds(15);
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

    private void StopMoving()
    {
        TestInputControl.MoveRight = false;
        TestInputControl.MoveLeft = false;
    }
}
```

Bonus Solution
 
`Measure.Custom("FPS", (int)(1f / Time.deltaTime));`
