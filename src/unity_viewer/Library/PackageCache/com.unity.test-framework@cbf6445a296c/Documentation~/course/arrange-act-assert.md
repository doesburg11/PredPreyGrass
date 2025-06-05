# 2\. Arrange, Act, Assert

## Learning Objectives

In this exercise, you will learn about the core unit testing principle of AAA (Arrange, Act, Assert), which will help you structure your unit test.

## Intro and Motivation

The Arrange, Act, Assert concept is an industry standard in unit testing. It allows for a clear distinction of the code for setting up your test, carrying out the test, and evaluation. Using this can make your test more readable both for yourself and for your colleagues.  
  
In the first part of the code, we arrange all the elements needed for the test. In the middle part, we act on the object that is under test. In the final part, we assert on the result of the act part. The three parts of the code are usually separated by an empty line.  
  
An example of Arrange, Act, Assert could look like the following:  
```
[Test]
public void StringWriterTest()
{
 // Arrange
 var stringWriterUnderTest = new StringWriter();
 stringWriterUnderTest.NewLine = "\\n";
 var testStringA = "I am testing";
 var testStringB = "with new line";

 // Act
 stringWriterUnderTest.WriteLine(testStringA);
 stringWriterUnderTest.WriteLine(testStringB);

 // Assert
 Assert.AreEqual("I am testing\\nwith new line\\n", stringWriterUnderTest.ToString());
}
```  
  
It is good practice to use `XUnderTest` as a variable name of the class that is being tested. This helps to keep the focus of the test clean.  

The Act part of the code should have as few lines as possible, reflecting what is actually being tested. The assert should in the optimal case only contain assert calls, but it can also be necessary to include some lines of logic to allow for the assertion.

## Exercise

Import the [sample](./welcome.md#import-samples) `2_ActArrangeAssert` into your Unity Editor (version 2019.2 or newer) from the Package Manager window.  

In this project we have a class called `StringFormatter`. It has two methods of interest: `void Configure(string joinDelimiter)` and `string Join(object[] args)`.  
  
The goal of this exercise is to write one or more tests, testing the `Join` method. For example, testing that it can join with a ";" (semicolon) delimiter.

## Hints

*   Setup of the test input and the call to `Configure(";")` would go into the `Arrange` part of your test.
*   It is good practice to separate the three parts of your test (arrange, act and assert) with a blank line.

## Solution

The exercise can be solved with a test like the following:

```
[Test]
public void JoinsObjectsWithSemiColon()
{
 // Arrange
 var formatterUnderTest = new StringFormatter();
 formatterUnderTest.Configure(";");
 var objects = new object[] {"a", "bc", 5, "d"};
 
 // Act
 var result = formatterUnderTest.Join(objects);
 
 // Assert
 Assert.AreEqual("a;bc;5;d", result);
}
```

A full project with the solution can be found in the sample `2_ActArrangeAssert_Solution.`

## Further reading and Resources

The AAA concept is a widely used standard which you can read more about in many online sources, including [this blogpost.](https://defragdev.com/blog/?p=783)
