# Exclude code from Burst compilation

By default, Burst compiles all methods in jobs decorated with the `[BurstCompile]` attribute. But some methods aren't appropriate for Burst compilation. For example, methods that perform logging using managed objects or that check the validity of something only valid in a managed environment can only run in a .NET runtime. In such cases you can use the [`[BurstDiscard]`](xref:Unity.Burst.BurstDiscardAttribute) attribute on a method or property to exclude it from Burst compilation:

```c#
[BurstCompile]
public struct MyJob : IJob
{
    public void Execute()
    {
        // Only executed when running from a full .NET runtime
        // this method call will be discard when compiling this job with
        // [BurstCompile] attribute
        MethodToDiscard();
    }

    [BurstDiscard]
    private static void MethodToDiscard(int arg)
    {
        Debug.Log($"This is a test: {arg}");
    }
}
```
>[!NOTE]
>A method with `[BurstDiscard]` can't have a return value.

You can use a `ref` or `out` parameter, which indicates whether the code is running on Burst or managed:

```c#
[BurstDiscard]
private static void SetIfManaged(ref bool b) => b = false;

private static bool IsBurst()
{
    var b = true;
    SetIfManaged(ref b);
    return b;
}
```

## Additional resources

* [`[BurstDiscard]` attribute API reference](xref:Unity.Burst.BurstDiscardAttribute)
* [Marking code for Burst compilation](compilation-burstcompile.md)
