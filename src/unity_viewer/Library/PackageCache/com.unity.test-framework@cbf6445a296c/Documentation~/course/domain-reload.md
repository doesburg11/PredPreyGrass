# 13\. Domain reload

## Learning objectives

In this section, you will learn how to invoke and wait for Domain Reloads.

## Intro and motivation

When performing actions that affect the scripts in a project, Unity performs a domain reload. Since a domain reload restarts all scripts, then it's necessary to mark any expected domain reload by yielding a [WaitForDomainReload](https://docs.unity3d.com/Packages/com.unity.test-framework@1.1/manual/reference-wait-for-domain-reload.html). The command stops any further code execution and then resumes after the domain reload is done.  
  
It's also possible to yield a [RecompileScripts](https://docs.unity3d.com/Packages/com.unity.test-framework@1.1/manual/reference-recompile-scripts.html) command. This does the same as `WaitForDomainReload` except that it performs an `AssetDatabase.Reload()` call. Both calls can be configured to expect whether a script compilation is expected to succeed.  
  
If a domain reload happens while a test is running without yielding one of these commands, then the test will fail with an error about an unexpected domain reload.

## Exercise

The [sample](./welcome.md#import-samples) `13_DomainReload_Solution` is set up with a test class called `ScriptAddingTests`.  
  
The test has two helper methods already implemented:  

*   `CreateScript` creates a C# script with a class called `MyTempScript`. That has a method called `Verify`.
  
*   `VerifyScript` instantiates an instance of `MyTempScript` using reflection and returns the value from the `Verify` method. The expected return value is the string "OK".

After running `CreateScript` Unity now has a new C# file in the project and thus needs to recompile. The task is to create a test that calls `CreateScript`, handles the domain reload and then verifies the output from `VerifyScript`.  
  
Remember that your script should also clean up after itself, by deleting the file and recompiling the script again. This is recommended to do in a `TearDown` or `UnityTearDown`, which will run even if the test fails.

> **Important**: After importing, you should **move the sample test folder** `Tests_13` into the `Assets` folder for this exercise to work.

## Hints

*   If `RecompileScripts` is unavailable to you due to it being internal, then you need to upgrade the Unity Test Framework package to version 1.1.0 or higher.
*   If you are on a non-Windows machine you might want to change paths inside **k\_fileName** or use C# [Path.Combine](https://docs.microsoft.com/en-us/dotnet/api/system.io.path.combine?view=net-6.0) for more cross-platform safe code.

## Solution

A full solution is available in the sample `13_DomainReload_Solution`.  
  
The test can be implemented as follows:

```
internal class ScriptAddingTests
{
 private const string k_fileName = @"Assets\\Tests\\TempScript.cs";
 
 [UnityTest]
 public IEnumerator CreatedScriptIsVerified()
 {
  CreateScript();
  yield return new RecompileScripts();

  var verification = VerifyScript();
  
  Assert.That(verification, Is.EqualTo("OK"));
 }

 [UnityTearDown]
 public IEnumerator Teardown()
 {
  if (!File.Exists(k_fileName))
  {
   yield break;
  }
  
  File.Delete(k_fileName);
  yield return new RecompileScripts();
 }
 
 private void CreateScript()
 {
  File.WriteAllText(k_fileName, @"
  public class MyTempScript {
   public string Verify()
   {
    return ""OK"";
   } 
  }");
 }

 private string VerifyScript()
 {
  Type type = Type.GetType("MyTempScript", true);
  
  object instance = Activator.CreateInstance(type);

  var verifyMethod = type.GetMethod("Verify", BindingFlags.Instance | BindingFlags.Public);

  var verifyResult = verifyMethod.Invoke(instance, new object[0]);
  return verifyResult as string;
 }
}
```

## Further reading and resources

[Documentation for RecompileScripts.](https://docs.unity3d.com/Packages/com.unity.test-framework@1.1/manual/reference-recompile-scripts.html)  
[Documentation for WaitForDomainReload.](https://docs.unity3d.com/Packages/com.unity.test-framework@1.1/manual/reference-wait-for-domain-reload.html)
