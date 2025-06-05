# 10\. Long running tests

## Learning objectives

This exercise will cover best practices and pitfalls for tests that have a long runtime, such as tests yielding back a `WaitForSeconds`.

## Intro and motivation

In Play Mode it is possible for UnityTests to return [Yield instructions](https://docs.unity3d.com/ScriptReference/YieldInstruction.html), such as `WaitForSeconds`. This is supported because in some test cases it can be valid to wait for a limited time. However, long-running tests are in general a bad practice that should be avoided when possible. If you can't avoid a long-running test, it's recommended to provide the test with `[Category]` and `[Explicit]` attributes. The `[Category]` attribute is used to label tests with a category name that can later be used as a filter to run a subset of tests selectively. The `[Explicit]` attribute ensures that the test is not run by default when running all tests. The test is only run when it is explicitly selected in the UI, or when its category is selected.  
  
```
[UnityTest]
[Explicit, Category("integration")]
public IEnumerator MySlowTest()
{
...
}
```  
  
In practice, this means that if you give some long-running tests the category "integration", then they will only be run if the "integration" category is selected. This makes it possible to keep "All tests" running relatively fast, even on a large project. It is also possible to specify the `[Explicit]` and `[Category]` attributes on a class level, which then applies to all tests in the class and on an assembly level, which applied to all tests inside that assembly. An example with it applied to assemblies:  

```
[assembly:Explicit]
[assembly:Category("integration")]
```

It is a good practice to have assembly level attributes defined in an `AssemblyInfo.cs` file.

## Exercise

Import the [sample](./welcome.md#import-samples) `10_LongRunningTests`, which is set up with a test assembly for Play Mode.  
  
The exercise is to add a new `UnityTest`, which yields back a `WaitForSeconds` command and then tag it accordingly with `[Category]` and `[Explicit]` tags.  
  
When pressing **RunAll**, the test should be skipped. When the Category is selected in the category drop down in the UI, then the test should not be skipped when **RunAll** is selected.

## Solution

The sample `10_LongRunningTests_Solution` contains the solution.  
  
The implemented test can look like this:

```
[UnityTest]
[Explicit, Category("integration")]
public IEnumerator ASlowTest()
{
 yield return new WaitForSeconds(5);
}
```

## Further reading and resources

[Nunit documentation for the Category attribute](https://docs.nunit.org/articles/nunit/writing-tests/attributes/category.html)
