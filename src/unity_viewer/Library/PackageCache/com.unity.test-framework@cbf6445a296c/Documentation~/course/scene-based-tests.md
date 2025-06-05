# 11\. Scene-based tests

## Learning objectives

In this exercise, you will learn how to test content that is stored in a scene.

## Intro and motivation

A useful scenario for our customers is using the test framework for verifying the content of a scene. That could be checking for certain GameObjects and MonoBehaviors.  
  
The [EditorSceneManager](https://docs.unity3d.com/ScriptReference/SceneManagement.EditorSceneManager.html) allows for loading and saving scenes. In combination with the test framework, this allows for the implementation of tests that verify a scene.  
  
When changing the state of the Editor in a test, such as loading a scene, it's good practice to clean up afterward. This can be done in a method with the `[TearDown]` attribute.

## Exercise

Import the [sample](./welcome.md#import-samples) `11_SceneBasedTests`, which contains a scene named `MyGameScene` and an assembly for Edit Mode tests.  
  
The task is to create a test that opens the scene, verifies that the scene contains a game object named `GameObjectToTestFor`.  
  
As cleanup, it should open a new empty scene, which is the default for Edit Mode tests. It is recommended to put that in a `[TearDown]`, which ensures that the cleanup code is run, even if the test fails.

## Hints

*   `EditorSceneManager.OpenScene("Assets\\MyGameScene.unity");` loads the scene
*   `EditorSceneManager.NewScene(NewSceneSetup.DefaultGameObjects, NewSceneMode.Single);` cleans up by changing back to an empty scene.

## Solution

A full solution is available in the sample `11_SceneBasedTests_Solution`.  
  
The test implementation can look like this:

```
public class SceneTests
{
 [SetUp]
 public void Setup()
 {
  EditorSceneManager.OpenScene("Assets\\MyGameScene.unity");
 }
 
 [Test]
 public void VerifyScene()
 {
  var gameObject = GameObject.Find("GameObjectToTestFor");
  
  Assert.That(gameObject, Is.Not.Null);
 }

 [TearDown]
 public void Teardown()
 {
  EditorSceneManager.NewScene(NewSceneSetup.DefaultGameObjects, NewSceneMode.Single);
 }
}
```

## Further reading and resources

[Documentation for EditorSceneManage api](https://docs.unity3d.com/ScriptReference/SceneManagement.EditorSceneManager.html)
