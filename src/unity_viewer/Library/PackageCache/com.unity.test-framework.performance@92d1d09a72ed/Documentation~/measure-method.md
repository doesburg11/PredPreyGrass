# Measure.Method()

Executes the provided method, sampling performance using the following additional properties/methods to control how the measurements are taken:
* **WarmupCount(int n)** - number of times to execute before measurements are collected. If unspecified, a default warmup is executed for 100 ms or until at least 3 method executions have completed, whichever is longer.
* **MeasurementCount(int n)** - number of measurements to capture, defaults to 9 if not specified.
* **DynamicMeasurementCount(OutlierMode outlierMode)** - dynamically find a suitable measurement count based on the margin of error of the samples. The measurements will stop once a certain amount of samples (specified by a confidence interval) falls within an acceptable error range from the result (defined by a relative error of the mean). A default margin of error range of 2% and a default confidence interval of 99% will be used. Statistical outliers will not be taken into account unless different behaviour is specified through the outlierMode argument.
* **DynamicMeasurementCount(double maxRelativeError, ConfidenceLevel confidenceLevel, OutlierMode outlierMode)** - dynamically find a suitable measurement count based on the margin of error of the samples and using the provided confidence interval and error range.
* **IterationsPerMeasurement(int n)** - number of method executions per measurement to use. If this value is not specified, the method is executed as many times as possible until approximately 100 ms has elapsed.
* **SampleGroup(string name)** - name of the measurement, defaults to "Time" if unspecified.
* **SampleGroup(SampleGroup sampleGroup)** - a sample group with a custom name and measurement unit. This will override the otherwise default value of "Time".
* **GC()** - if specified, measures the total number of Garbage Collection allocation calls.
* **SetUp(Action action)** - is called every iteration before method execution. Setup time is not measured.
* **CleanUp(Action action)** - is called every iteration after method execution. Cleanup time is not measured.


#### Example 1: Simple method measurement using default values

``` csharp
[Test, Performance]
public void Test()
{
    Measure.Method(() => { ... }).Run();
}
```

#### Example 2: Customize Measure.Method properties

``` csharp
[Test, Performance]
public void Test()
{
    Measure.Method(() => { ... })
        .WarmupCount(10)
        .MeasurementCount(10)
        .IterationsPerMeasurement(5)
        .GC()
        .SetUp(()=> {/*setup code*/})
        .CleanUp(()=> {/*cleanup code*/})
        .Run();
}
```

#### Example 3: Measure.Method, SampleGroup and ProfilerMarkers

``` csharp
[Test, Performance]
public void Test()
{
    var sg = new SampleGroup("Name", SampleUnit.Microsecond);
    var sgMarker = new SampleGroup("MarkerName", SampleUnit.Seconds);
    Measure.Method(() => { ... }).SampleGroup(sg).ProfilerMarkers(sgMarker).Run();
}
```