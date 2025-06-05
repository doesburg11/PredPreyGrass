# Measure.ProfilerMarkers()

Used to record profiler markers. Profiler marker timings are recorded automatically and sampled within the scope of the `using` statement. Names should match profiler marker labels. Profiler markers are sampled once per frame. Sampling the same profiler marker per frame will result in the sum of all invocations. 

You can also create your own SampleGroups, specifying a custom name and the measurement units you want your results in, see [example 2](#example-2-measuring-profiler-markers-in-a-scope-with-custom-samplegroups). 

## Limitations

* Not supported in Unity Test Framework [Edit Mode tests](https://docs.unity3d.com/Packages/com.unity.test-framework@latest?subfolder=/manual/edit-mode-vs-play-mode-tests.html#edit-mode-tests).
* Not supported in the Unity Profiler's [Deep Profiling](https://docs.unity3d.com/Manual/ProfilerWindow.html#deep-profiling) mode.
* Profiler markers created using `Profiler.BeginSample()` are not supported, switch to `ProfilerMarker` if possible.

#### Example: Measuring profiler markers in a scope

``` csharp
[Test, Performance]
public void Test()
{
    string[] markers =
    {
        "Instantiate",
        "Instantiate.Copy",
        "Instantiate.Produce",
        "Instantiate.Awake"
    };

    using(Measure.ProfilerMarkers(markers))
    {
        ...
    }
}
```

#### Example 2: Measuring profiler markers in a scope with custom SampleGroups

``` csharp

[UnityTest, Performance]
public IEnumerator Test()
{
    var sampleGroups = new []{
        new SampleGroup("Instantiate", SampleUnit.Second), 
        new SampleGroup("Instantiate.Copy", SampleUnit.Nanosecond),
        new SampleGroup("Instantiate.Awake", SampleUnit.Microsecond)
    };

    using (Measure.ProfilerMarkers(sampleGroups))
    {
        ...
    }
}
```