# Measure.Custom()

When you want to record samples outside of frame time, method time, or profiler markers, use a custom measurement. It can be any value.

#### Example: Use a custom measurement to capture total allocated memory

``` csharp
[Test, Performance]
public void Test()
{
    SampleGroup sampleGroup = new SampleGroup("TotalAllocatedMemory", SampleUnit.Megabyte, false);
    var allocatedMemory = UnityEngine.Profiling.Profiler.GetTotalAllocatedMemoryLong() / 1048576f;
    Measure.Custom(sampleGroup, allocatedMemory);
}
```
