# 7\. Scene Validation Test

## Learning Objectives

Test scene for presence of Sara and Wand game object. Utilize Test Framework feature to make this test use all scenes as fixtures.

## Exercise

1.  Create **ValidationTest.cs** file with a single namespace and two classes _SceneValidationTests_ and _GameplayScenesProvider_.
2.  In the Tests class create **SaraAndWandArePresent** test to check that "Sara Variant" and "Wand" game objects are not null.
3.  In the Fixture class `GameplayScenesProvider` implement `IEnumerable<string>` and in generator method yield all scenes from [EditorBuildSettings.scenes](https://docs.unity3d.com/ScriptReference/EditorBuildSettings-scenes.html).
4.  Use `TestFixture` and [TestFixtureSource](https://docs.nunit.org/articles/nunit/writing-tests/attributes/testfixturesource.html) annotations on _SceneValidationTests_ class.
5.  Create a new Empty Scene and attach it to `EditorBuildSettings` to verify if tests are created dynamically.

## Hints

*   `TestFixture` and `TestFixtureSource` NUnit annotations require Test Class to be present inside Namespace.
*   To attach a scene to `EditorBuildSettings`, you need to create a new Scene, and then add it to **File > Build Settings**.

## Solution

ValidationTests.cs 
```
using System.Collections;
using System.Collections.Generic;
using NUnit.Framework;
using UnityEditor;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.TestTools;

namespace ValidationTests
{
    [TestFixture]
    [TestFixtureSource(typeof(GameplayScenesProvider))]
    public class SceneValidationTests
    {
        private readonly string _scenePath;
    
        public SceneValidationTests(string scenePath)
        {
            _scenePath = scenePath;
        }
        
        [OneTimeSetUp]
        public void LoadScene()
        {
            SceneManager.LoadScene(_scenePath);
        }
        
        [UnityTest]
        public IEnumerator SaraAndWandArePresent()
        {
            yield return waitForSceneLoad();
            var wand = GameObject.Find("Wand");
            var sara = GameObject.Find("Sara Variant");
            
            Assert.NotNull(wand, "Wand object exists");
            Assert.NotNull(sara, "Sara object exists");
        }
        
        IEnumerator waitForSceneLoad()
        {
            while (!SceneManager.GetActiveScene().isLoaded)
            {
                yield return null;
            }
        }
    }
    
    public class GameplayScenesProvider : IEnumerable
    {
        public IEnumerator GetEnumerator()
        {
            foreach (var scene in EditorBuildSettings.scenes)
            {
                if (!scene.enabled || scene.path == null)
                {
                    continue;
                }
    
                yield return scene.path;
            }
        }
    
        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }
    }
}
```
