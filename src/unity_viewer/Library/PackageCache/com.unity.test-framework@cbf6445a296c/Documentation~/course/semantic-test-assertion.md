# 3\. Semantic test assertion

## Learning objectives

This exercise introduces the `Assert.That` and related classes.

## Intro and motivation

The NUnit test framework and the Unity Test Framework have a series of classes for asserting objects in a way that is closer to natural language. This makes the statements easily readable.  

Here are some examples on how to use the semantic assertion classes:

```
Assert.That(myValue, Is.GreaterThan(20));
Assert.That(str, Does.Contain("a string").And.Contain("something else"));
``` 

Here we check that the variable `myValue` is greater than 20 and then that the string `str` contains both "a string" and "something else".  
  
The semantic assertion is also known as [Constraint Model](https://docs.nunit.org/articles/nunit/writing-tests/assertions/assertion-models/constraint.html). Other than `It` and `Does` there are multiple other keywords that can be used.

## Exercise

In the `3_SemanticTestAssertion` [sample](./welcome.md#import-samples), there is a class called `ValueOutputter`, which returns values of different types.  
  
Write tests that assert on the different outputs. It should be verified that:

*   `GetInt()` returns 11.
*   `GetString()` returns a string that contains the words `string` and `asserted`.
*   `GetFloat()` returns a value that is around 19.33.

## Hints

*   Asserting on the float might require a check for the value being greater than 19.33 and less than 19.34, as the output is not rational.

## Solution

A full solution to the exercise is available in the sample `3_SemanticTestAssertion_Solution`.  
  
```
internal class ValueOutputterTests
{
 [Test]
 public void GivesExpectedInt()
 {
  var outputterUnderTest = new ValueOutputter();

  var number = outputterUnderTest.GetInt();
  
  Assert.That(number, Is.EqualTo(11));
 }
 
 [Test]
 public void GivesExpectedString()
 {
  var outputterUnderTest = new ValueOutputter();

  var str = outputterUnderTest.GetString();
  
  Assert.That(str, Does.Contain("string").And.Contain("asserted"));
 }
 
 [Test]
 public void GivesExpectedFloat()
 {
  var outputterUnderTest = new ValueOutputter();

  var number = outputterUnderTest.GetFloat();
  
  Assert.That(number, Is.GreaterThan(19.33f).And.LessThan(19.34f));
 }
}
```

## Further reading and resources

[NUnit 2 documentation for the constraint model](https://nunit.org/docs/2.4/constraintModel.html)
