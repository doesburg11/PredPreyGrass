# 15\. Test cases

## Learning objectives

This section will cover `[TestCase]` and similar NUnit attributes and how to work with them in UnityTests.

## Intro and motivation

NUnit has a few tools for parameterized tests, which can be used to specify test cases with variating parameters. This can drastically reduce the amount of repeated code and make the test cleaner to use.  
  
An example of a parameterized test using the `[TestCase]` attribute:  

```
[Test]
[TestCase(49, "a string", true)]
[TestCase(9, "something", false)]
public void MyTest(int firstValue, string secondValue, bool expectedOutcome)
{
 ...
}  
```

This will generate two tests, each with a different input to the method body.  
  
In addition to the `[TestCase]` attribute, NUnit also has a `[Values]` attribute, which specifies a set of values on each individual input. An example of such is:  

```
[Test]
public void MyTest([Values(49, 9)]int firstValue, [Values("a string", "something")]string secondValue)
{
 ...
}  
```

When specifying multiple input parameters, they are treated as combinatorial. That means that each combination of them will be tested. For the above example, that will result in a total of 4 cases:  

```
MyTest(49, "a string")
MyTest(49, "something")
MyTest(9, "a string")
MyTest(9, "something") 
```

This can easily explode into many combinations. The combinations might not all be valuable and would just waste time, so use this with care.  
  
For both the `[TestCase]` and `[Values]` attributes, there is a more dynamic version called `[TestCaseSource]` and `[ValueSource]` accordingly. These each take in a static method or array, returning a collection of objects.  
  
Of these 4 methods, the `[ValueSource]` attribute is currently the only one supported by `[UnityTest]`. Since this would produce combinational tests, if multiple arguments with `[ValueSource]` are provided, then it is recommended to make a test case struct, if multiple arguments are needed for the test. An example of such could look like this:  

```
[UnityTest]
public IEnumerator AddAsyncCalculatesCorrectValue([ValueSource(nameof(TestCases))] TestCase testCase)
{
 ...
}

private static IEnumerable TestCases()
{
 yield return new TestCase {value1 = 4, value2 = "a string"};
 yield return new TestCase {value1 = 8, value2 = "another string"};
}

public struct TestCase
{
 public int value1;
 public string value2;

 public override string ToString()
 {
  return $"{value1}, {value2}";
 }
}
```

## Exercise

In the [sample](./welcome.md#import-samples) `15_TestCases` a class is set up with some basic math. It has two methods:  

*   `Add` which takes two integers and adds them together.
  
*   `AddAsync` also adds two integers together, but does so asynchronously, yielding back an `IEnumerator`

The task is to add tests for the two methods. The `AddAsync` method first returns the result after a few frames, so that will be best suited for a `[UnityTest]`. Note that it is not enough to yield back the `IEnumerator`, as the test framework does not curently support nested yields. Instead, create a loop to move over each element until it's done. At each step of the while loop, let the test yield back null.  

## Hints

*   The `ToString()` implementation in the struct is there to provide readable info in the test runner treeview. Without it, it would just show the struct name as the test argument for every case.

## Solution

A solution for the exercise is available in the sample `15_TestCases_Solution`. Tests for both methods can be implemented as follows:  

```
[Test]
[TestCase(24, 80, 104)]
[TestCase(10, -15, -5)]
[TestCase(int.MaxValue, 10, int.MinValue + 9)]
public void AddCalculatesCorrectValue(int valueA, int valueB, int expectedResult)
{
 var myClass = new MyClass();

 var result = myClass.Add(valueA, valueB);
 
 Assert.That(result, Is.EqualTo(expectedResult));
}

[UnityTest]
public IEnumerator AddAsyncCalculatesCorrectValue([ValueSource(nameof(AdditionCases))] AddCase addCase)
{
 var myClass = new MyClass();

 var enumerator = myClass.AddAsync(addCase.valueA, addCase.valueB);
 while (enumerator.MoveNext())
 {
  yield return null;
 }
 var result = enumerator.Current;
 
 Assert.That(result, Is.EqualTo(addCase.expectedResult));
}

private static IEnumerable AdditionCases()
{
 yield return new AddCase {valueA = 24, valueB = 80, expectedResult = 104};
 yield return new AddCase {valueA = 10, valueB = -15, expectedResult = -5};
 yield return new AddCase {valueA = int.MaxValue, valueB = 10, expectedResult = int.MinValue + 9};
}

public struct AddCase
{
 public int valueA;
 public int valueB;
 public int expectedResult;

 public override string ToString()
 {
  return $"{valueA} + {valueB} = {expectedResult}";
 }
}
```
