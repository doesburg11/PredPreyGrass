using UnityEngine;
using UnityEngine.TestTools;
using NUnit.Framework;
using System.Collections;

internal class NewPlayModeTest
{
    //This script is just to have a referance to nunit
    [Test]
    public void NewPlayModeTestSimplePasses()
    {
    }

    [UnityTest]
    public IEnumerator NewPlayModeTestWithEnumeratorPasses()
    {
        yield return null;
    }
}
