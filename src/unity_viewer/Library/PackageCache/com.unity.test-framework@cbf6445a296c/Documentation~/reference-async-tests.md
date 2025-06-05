# Async tests

You can use the dotnet Task asynchronous programming model to write asynchronous tests. If you're new to asynchronous programming and its applications, see the [Microsoft documentation](https://docs.microsoft.com/en-us/dotnet/csharp/programming-guide/concepts/async/) for a comprehensive guide. See also the documentation for NUnit [Test](https://docs.nunit.org/articles/nunit/writing-tests/attributes/test.html), which explains the requirements for an async test. Async code is run on the main thread and Unity Test Framework will `await` it by checking if the task is done on each [update](https://docs.unity3d.com/ScriptReference/PlayerLoop.Update.html) for Play Mode or on each [EditorApplication.update](https://docs.unity3d.com/ScriptReference/EditorApplication-update.html) outside Play Mode.

Note that any failing log messages will first be evaluated after the async test has completed. This means that if you have a failing log message in an async test, it will not be reported until the test has completed.

The following code snippet demonstrates an async test based on Microsoft's making breakfast example. Note that the test method is marked with the `async` keyword and has return type `Task`. We set up a list of Tasks corresponding to asynchronous methods representing parts of the breakfast making process. We use `await` to start these tasks in a non-blocking way, write to the log when each one completes, and write again to the log when all are completed.

```
using System.Collections.Generic;
using System.Threading.Tasks;
using NUnit.Framework;
using UnityEngine;

public class AsyncExample
{
    [Test]
    public async Task MakeBreakfast_InTheMorning_IsEdible()
    {
        var eggsTask = FryEggsAsync(2);
        var baconTask = FryBaconAsync(3);
        var toastTask = MakeToastWithButterAndJamAsync(2);

        var breakfastTasks = new List<Task> { eggsTask, baconTask, toastTask };
        while (breakfastTasks.Count > 0)
        {
            Task finishedTask = await Task.WhenAny(breakfastTasks);
            if (finishedTask == eggsTask)
            {
                Debug.Log("eggs are ready");
            }
            else if (finishedTask == baconTask)
            {
                Debug.Log("bacon is ready");
            }
            else if (finishedTask == toastTask)
            {
                Debug.Log("toast is ready");
            }
            breakfastTasks.Remove(finishedTask);
        }

        Debug.Log("Breakfast is ready!");
    }

    static async Task<Toast> MakeToastWithButterAndJamAsync(int number)
    {
        var toast = await ToastBreadAsync(number);
        ApplyButter(toast);
        ApplyJam(toast);

        return toast;
    }

    private static Juice PourOJ()
    {
        Debug.Log("Pouring orange juice");
        return new Juice();
    }

    private static void ApplyJam(Toast toast) =>
        Debug.Log("Putting jam on the toast");

    private static void ApplyButter(Toast toast) =>
        Debug.Log("Putting butter on the toast");

    private static async Task<Toast> ToastBreadAsync(int slices)
    {
        for (int slice = 0; slice < slices; slice++)
        {
            Debug.Log("Putting a slice of bread in the toaster");
        }
        Debug.Log("Start toasting...");
        await Task.Delay(3000);
        Debug.Log("Remove toast from toaster");

        return new Toast();
    }

    private static async Task<Bacon> FryBaconAsync(int slices)
    {
        Debug.Log($"putting {slices} slices of bacon in the pan");
        Debug.Log("cooking first side of bacon...");
        await Task.Delay(3000);
        for (int slice = 0; slice < slices; slice++)
        {
            Debug.Log("flipping a slice of bacon");
        }
        Debug.Log("cooking the second side of bacon...");
        await Task.Delay(3000);
        Debug.Log("Put bacon on plate");

        return new Bacon();
    }

    private static async Task<Egg> FryEggsAsync(int howMany)
    {
        Debug.Log("Warming the egg pan...");
        await Task.Delay(3000);
        Debug.Log($"cracking {howMany} eggs");
        Debug.Log("cooking the eggs ...");
        await Task.Delay(3000);
        Debug.Log("Put eggs on plate");

        return new Egg();
    }

    private static Coffee PourCoffee()
    {
        Debug.Log("Pouring coffee");
        return new Coffee();
    }

    public struct Toast
    {

    }

    public struct Juice
    {

    }

    public struct Bacon
    {

    }

    public struct Egg
    {

    }

    public struct Coffee
    {

    }
}
```

The following shows the result of running this example in the **Test Runner** window:

![Run async test example](./images/async.png)

## Editor freezing on `Assert.ThrowsAsync` and workaround

NUnit's assertion for asynchronous code, [Assert.ThrowsAsync](https://docs.nunit.org/articles/nunit/writing-tests/assertions/classic-assertions/Assert.ThrowsAsync.html), blocks the calling thread until the async function you pass in completes. By default Unity runs asynchronous functions on the main thread in case they need to call the Editor API, which means `Assert.ThrowsAsync` can lock up the main thread and cause the Editor to freeze.

To workaround this problem, you can unwrap the `Assert.ThrowsAsync` logic into your own `try`/`catch` blocks and assert that you caught something. For example, **do** this:

```
[Test]
public async Task ThisDoesNotLockTheMainThread()
 {
  bool caught = false;
  try
  { 
    await Task.Delay(1000); throw new System.Exception("Hello world."); }
    catch (System.Exception x)
   {
     caught = true;
   }
   Assert.That(caught);
}
```

**Instead of** this:

```
[Test]
public void ThisLocksTheMainThread()
 {
  Assert.ThrowsAsync<System.Exception>(async () =>
   { await Task.Delay(1000); throw new System.Exception("Hello world."); }
  );
}
```
