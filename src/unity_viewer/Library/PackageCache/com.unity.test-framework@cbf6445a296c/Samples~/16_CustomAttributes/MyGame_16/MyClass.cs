using System.Collections;
using System.Collections.Generic;
using System.Threading;
using UnityEngine;

public class MyClass
{
    public bool DoSomething()
    {
        return true;
    }

    public bool DoSomethingElse()
    {
        // here is a regression that somebody made. It seems slow
        Thread.Sleep(500);

        return true;
    }
}
