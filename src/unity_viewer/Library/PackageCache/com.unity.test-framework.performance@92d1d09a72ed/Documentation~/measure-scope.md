# Measure.Scope(string name = "Time")

Measures execution time for the scope as a single time, for both synchronous and coroutine methods. Passing the name argument overrides the name of the created SampleGroup.
The defualt SampleGroup is named "Time" and with Milliseconds as measurement unit. You can also create your own SampleGroup, specifying a custom name and the measurement unit you want your results in, see [example 2](#example-2-specify-custom-samplegroup).

#### Example 1: Measuring a scope; execution time is measured for everything in the using statement

``` csharp
[Test, Performance]
public void Test()
{
    using(Measure.Scope())
    {
        ...
    }
}
```

#### Example 2: Specify Custom SampleGroup

``` csharp
[Test, Performance]
public void Test()
{
    var sampleGroup = new SampleGroup("Scope", SampleUnit.Microsecond);
    using (Measure.Scope(sg))
    {
        ...
    }
}
```