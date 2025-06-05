# Burst compilation in Play mode

When you create a [build of your project](xref:um-build-types), Burst compiles all the supported code ahead-of-time (AOT) into a native library which Unity ships with your application.

When previewing your application in the Editor's Play mode, Burst provides the following compilation modes:

* Asynchronous: Parts of your code marked for Burst compilation can run as managed, just-in-time (JIT) compiled code in the .NET runtime while waiting for Burst compilation to complete. This is the default behavior.
* Synchronous: Parts of your code marked for Burst compilation can only run as Burst-compiled native code and your application must wait for Burst compilation to complete.

## Synchronous compilation

To force synchronous compilation in Play mode, set the [`CompileSynchronously`](xref:Unity.Burst.BurstCompileAttribute.CompileSynchronously) property to `true` as follows:

```c#
[BurstCompile(CompileSynchronously = true)]
public struct MyJob : IJob
{
    // ...
}
```

Waiting for synchronous compilation affects the current running frame, which can cause hitches and make your application unresponsive. Synchronous compilation is only recommended in the following situations:

* If you have a long running job that only runs once. The performance of the compiled code might outweigh the downsides of synchronous compilation.
* If you're profiling a Burst job and want to test the code from the Burst compiler. When you do this, perform a warmup to discard any timing measurements from the first call to the job. This is because the profiling data includes the compilation time and skews the result.
* To aid with debugging the difference between managed and Burst compiled code.

## Additional resources

* [Marking code for Burst compilation](compilation-burstcompile.md)
* [Generic jobs](compilation-generic-jobs.md)