# Measure.Frames()

Records time per frame by default and provides additional properties/methods to control how the measurements are taken:
* **WarmupCount(int n)** - number of times to execute before measurements are collected. If unspecified, a default warmup is executed for 80 ms or until at least 3 full frames have rendered, whichever is longest.
* **MeasurementCount(int n)** - number of frames to capture measurements for. If this value is not specified, as many frames as possible are captured until approximately 500 ms has elapsed.
* **DynamicMeasurementCount(OutlierMode outlierMode)** - dynamically find a suitable measurement count based on the margin of error of the samples. The measurements will stop once a certain amount of samples (specified by a confidence interval) falls within an acceptable error range from the result (defined by a relative error of the mean). A default margin of error range of 2% and a default confidence interval of 99% will be used. Statistical outliers will not be taken into account unless different behaviour is specified through the outlierMode argument.
* **DynamicMeasurementCount(double maxRelativeError, ConfidenceLevel confidenceLevel, OutlierMode outlierMode)** - dynamically find a suitable measurement count based on the margin of error of the samples and using the provided confidence interval and error range.
* **DontRecordFrametime()** - disables frametime measurement.
* **ProfilerMarkers(...)** - sample profile markers per frame. Does not work for deep profiling and `Profiler.BeginSample()`
* **SampleGroup(string name)** - name of the measurement, defaults to "Time" if unspecified.
* **Scope()** - measures frame times in a given coroutine scope. By default it uses a SampleGroup named "Time" with Milliseconds as measurement unit. You can also create your own SampleGroup, specifying a custom name and the measurement unit you want your results in, see [example 5](#example-5-specify-custom-samplegroup-in-the-scope).

## Limitations

* Not supported in Unity Test Framework [Edit Mode tests](https://docs.unity3d.com/Packages/com.unity.test-framework@latest?subfolder=/manual/edit-mode-vs-play-mode-tests.html#edit-mode-tests).

#### Example 1: Simple frame time measurement using default values of at least 7 frames and default WarmupCount (see description above).

``` csharp
[UnityTest, Performance]
public IEnumerator Test()
{
    ...

    yield return Measure.Frames().Run();
}
```

#### Example 2: Sample profile markers per frame, disable frametime measurement

If you'd like to sample profiler markers across multiple frames and don't need to record frametime, it is possible to disable the frame time measurement.

``` csharp
[UnityTest, Performance]
public IEnumerator Test()
{
    ...

    yield return Measure.Frames()
        .ProfilerMarkers(...)
        .DontRecordFrametime()
        .Run();
}
```

#### Example 3: Sample frame times in a scope

``` csharp
[UnityTest, Performance]
public IEnumerator Test()
{
    using (Measure.Frames().Scope())
    {
        yield return ...;
    }
}
```

#### Example 4: Specify custom WarmupCount and MeasurementCount per frame

If you want more control, you can specify how many frames you want to measure.

``` csharp
[UnityTest, Performance]
public IEnumerator Test()
{
    ...

    yield return Measure.Frames()
        .WarmupCount(5)
        .MeasurementCount(10)
        .Run();
}
```
#### Example 5: Specify Custom SampleGroup in the Scope

``` csharp
[UnityTest, Performance]
public IEnumerator Test()
{
    var sg = new SampleGroup("MarkerName", SampleUnit.Second);
    using (Measure.Frames().Scope(sg))
    {
        yield return ...;
    }
}
```

#### Example 6: Sample profile markers per frame with custom SampleGroups that change sample unit

``` csharp
[UnityTest, Performance]
public IEnumerator Test()
{
    var sampleGroup = new SampleGroup("Name", SampleUnit.Milliseconds);
    var profileMarkersSampleGroups = new []{
        new SampleGroup("MarkerName", SampleUnit.Second), 
        new SampleGroup("MarkerName1", SampleUnit.Nanosecond)
    };

    yield return Measure.Frames().SampleGroup(sg).ProfilerMarkers(profileMarkersSampleGroups).Run();

     ...
}
```