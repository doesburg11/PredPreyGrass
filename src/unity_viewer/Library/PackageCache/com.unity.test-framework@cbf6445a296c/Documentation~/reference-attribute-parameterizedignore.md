# ParametrizedIgnore attribute

This attribute is an alternative to the standard `Ignore` attribute in [NUnit](http://www.nunit.org/). It allows for ignoring tests based on arguments which were passed to the test method.

## Example

The following example shows a method to use the `ParametrizedIgnore` attribute to ignore only one test with specific combination of arguments, where someString is `b` and someInt is `10`.

```C#
using NUnit.Framework;
using System.Collections.Generic;
using UnityEngine.TestTools;

public class MyTestsClass
{
    public static IEnumerable<TestCaseData> MyTestCaseSource()
    {
        for (int i = 0; i < 5; i++)
        {
            yield return new TestCaseData("a", i);
            yield return new TestCaseData("b", i);
        }
    }

    [Test, TestCaseSource("MyTestCaseSource")]
    [ParametrizedIgnore("b", 3)]
    public void Test(string someString, int someInt)
    {
        Assert.Pass();
    }
}

```

It could also be used together with `Values` attribute in [NUnit](http://www.nunit.org/).

```C#
using NUnit.Framework;
using UnityEngine.TestTools;

public class MyTestsClass
{
    [Test]
    [ParametrizedIgnore("b", 10)]
    public void Test(
        [Values("a", "b")] string someString,
        [Values(5, 10)] int someInt)
    {
        Assert.Pass();
    }
}

```
