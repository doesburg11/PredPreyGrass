using System.Collections;
using System.Collections.Generic;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;

namespace PlayModeTests_10s
{
    public class SlowTests
    {
        [UnityTest]
        [Explicit, Category("integration")]
        public IEnumerator ASlowTest()
        {
            yield return new WaitForSeconds(5);
        }
    }
}
