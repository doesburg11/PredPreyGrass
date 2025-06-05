# Get started

You can use Burst to compile [jobs](https://docs.unity3d.com/ScriptReference/Unity.Jobs.IJob.html) or static methods in non-job C# types. To start using the Burst compiler in your code, decorate a job or static method with the [`[BurstCompile]`](xref:Unity.Burst.BurstCompileAttribute) attribute.

For more information on where and when to apply the `[BurstCompile]` attribute, refer to [Marking code for Burst compilation](compilation-burstcompile.md).

## Compiling jobs with Burst

For jobs, you only need to apply the `[BurstCompile]` attribute to the job declaration and Burst compiles everything inside the job automatically. The following example demonstrates this:

```c#
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using UnityEngine;

public class MyBurst2Behavior : MonoBehaviour
{
    void Start()
    {
        var input = new NativeArray<float>(10, Allocator.Persistent);
        var output = new NativeArray<float>(1, Allocator.Persistent);
        for (int i = 0; i < input.Length; i++)
            input[i] = 1.0f * i;

        var job = new MyJob
        {
            Input = input,
            Output = output
        };
        job.Schedule().Complete();

        Debug.Log("The result of the sum is: " + output[0]);
        input.Dispose();
        output.Dispose();
    }

    // Using BurstCompile to compile a Job with Burst

    [BurstCompile]
    private struct MyJob : IJob
    {
        [ReadOnly]
        public NativeArray<float> Input;

        [WriteOnly]
        public NativeArray<float> Output;

        public void Execute()
        {
            float result = 0.0f;
            for (int i = 0; i < Input.Length; i++)
            {
                result += Input[i];
            }
            Output[0] = result;
        }
    }
}
```

## Compiling static methods with Burst

For static methods, you must apply the `[BurstCompile]` attribute to both the individual methods you want Burst to compile and to the declaration of the parent type. The following example demonstrates this:

```c#
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using UnityEngine;

[BurstCompile]
public static class MyBurstUtilityClass
{
    [BurstCompile]
    public static void BurstCompiled_MultiplyAdd(in float4 mula, in float4 mulb, in float4 add, out float4 result)
    {
        result = mula * mulb + add;
    }
}
```

For more information on how you can call this Burst-compiled utility class and its member method from your C# code, refer to [Calling Burst-compiled code](csharp-calling-burst-code.md).

## Limitations

Burst supports most C# expressions and statements, with a few exceptions. For more information, refer to [C# language support](csharp-language-support.md).

## Compilation

Burst compiles your code [just-in-time (JIT)](https://en.wikipedia.org/wiki/Just-in-time_compilation) while in Play mode in the Editor, and [ahead-of-time (AOT)](https://en.wikipedia.org/wiki/Ahead-of-time_compilation) when your application runs in a Player. For more information on compilation, refer to [Burst compilation](compilation.md)

## Command line options

You can pass the following options to the Unity Editor on the command line to control Burst:

- `--burst-disable-compilation` disables Burst.
- `--burst-force-sync-compilation` force Burst to compile synchronously.

For more information, refer to [Burst compilation](compilation.md).

## Additional resources

* [Burst compilation](compilation.md)
* [`[BurstCompile]` attribute](compilation-burstcompile.md)
