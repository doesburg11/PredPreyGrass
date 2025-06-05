# 6\. SetUp and TearDown

## Learning objectives

In this exercise, you will get practical experience in using the NUnit attributes `[SetUp]` and `[TearDown]` in order to reduce code duplication in your tests.

## Intro and motivation

It's good practice to always let your test code clean up after itself and you also often need to set things up before running a test. If you have multiple tests, then that can easily become a lot of code duplication and if your test fails, your cleanup might not even be run, if you have not wrapped it in `try` and `finally` blocks.  
  
As a solution to this, NUnit has the `[SetUp]` and `[TearDown]` attributes. Methods with this attribute will be run before and after any of the classes respectively. If you are running multiple tests in your class at once, then the teardown and setup are run in between each of the tests.  

```
public class TestClass
{
 [SetUp]
 public void MySetUp() { ... }

 [Test]
 public void MyFirstTest() { ... }

 [Test]
 public void MySecondTest() { ... }

 [TearDown]
 public void MyTearDown() { ... }
}
```

## Exercise

Import the [sample](./welcome.md#import-samples) `6_SetUpTearDown`.  
  
In this project there is a class called `FileCreator`. It has two methods:  

*   `CreateEmptyFile(fileName)` - Creates an empty file in an `OutputFiles` directory
  
*   `CreateFile(string fileName, string content)` - Creates a file with the given content in an `OutputFiles` directory

  The catch is that it will throw a `DirectoryNotFoundException`, if there is no output called `OutputFiles` in the current directory. You will need to create this directory inside a `SetUp` method and remove it again afterwards with a `TearDown`. Your test can then assume that it starts with an emtpy directory, which simplifies the assertion.

## Hints

*   You can use `Directory.CreateDirectory` to create a directory.
*   You can use `Directory.Delete` with the recursive flag (second argument) set to delete the directory along with all its files.
*   `Directory.GetFiles` can be used to get files in a given directory.
*   `Path.Combine` is a handy method for combining parts of a file path. For example the directory name and the file name.

## Solution

The exercise can be solved with a test like the following:

```
[SetUp]
public void Setup()
{
 Directory.CreateDirectory(FileCreator.k_Directory);
}

[Test]
public void CreatesEmptyFile()
{
 var fileCreatorUnderTest = new FileCreator();
 var expectedFileName = "MyEmptyFile.txt";
 
 fileCreatorUnderTest.CreateEmptyFile(expectedFileName);

 var files = Directory.GetFiles(FileCreator.k_Directory);
 Assert.That(files.Length, Is.EqualTo(1), "Expected one file.");
 var expectedFilePath = Path.Combine(FileCreator.k_Directory, expectedFileName);
 Assert.That(files[0], Is.EqualTo(expectedFilePath));
}

[Test]
public void CreatesFile()
{
 var fileCreatorUnderTest = new FileCreator();
 var expectedFileName = "MyFile.txt";
 var expectedContent = "TheFileContent";
 
 fileCreatorUnderTest.CreateFile(expectedFileName, expectedContent);

 var files = Directory.GetFiles(FileCreator.k_Directory);
 Assert.That(files.Length, Is.EqualTo(1), "Expected one file.");
 var expectedFilePath = Path.Combine(FileCreator.k_Directory, expectedFileName);
 Assert.That(files[0], Is.EqualTo(expectedFilePath));
 var content = File.ReadAllText(expectedFilePath);
 Assert.That(content, Is.EqualTo(expectedContent));
}

[TearDown]
public void Teardown()
{
 Directory.Delete(FileCreator.k_Directory, true);
}
```

A full project with the solution can be found in the sample `6_SetUpTearDown.`

## Further reading and resources

[Nunit documentation for SetUp and TearDown](https://docs.nunit.org/articles/nunit/technical-notes/usage/SetUp-and-TearDown.html)
