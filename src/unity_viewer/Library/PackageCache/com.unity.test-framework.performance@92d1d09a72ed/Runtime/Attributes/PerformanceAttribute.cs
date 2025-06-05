using System;
using System.Collections;
using NUnit.Framework;
using NUnit.Framework.Interfaces;
using Unity.PerformanceTesting.Data;
using Unity.PerformanceTesting.Runtime;
using UnityEngine.TestTools;

namespace Unity.PerformanceTesting
{
    /// <summary>
    /// Test attribute to specify a performance test. It will add category "Performance" to test properties.
    /// </summary>
    [AttributeUsage(AttributeTargets.Method, AllowMultiple = false)]
    public class PerformanceAttribute : CategoryAttribute, IOuterUnityTestAction
    {
        /// <summary>
        /// Adds performance attribute to a test method.
        /// </summary>
        public PerformanceAttribute()
            : base("Performance") { }

        /// <summary>
        /// Executed before a test execution.
        /// </summary>
        /// <param name="test">Test to execute.</param>
        /// <returns>Enumerable collection of actions to perform before test setup.</returns>
        public IEnumerator BeforeTest(ITest test)
        {
            if (RunSettings.Instance == null)
            {
                RunSettings.Instance = ResourcesLoader.Load<RunSettings>(Utils.RunSettings, Utils.PlayerPrefKeySettingsJSON);
            }

            // domain reload will cause this method to be hit multiple times
            // active performance test is serialized and survives reloads
            if (PerformanceTest.Active == null)
            {
                PerformanceTest.StartTest(test);
                yield return null;
            }
        }

        /// <summary>
        /// Executed after a test execution.
        /// </summary>
        /// <param name="test">Executed test.</param>
        /// <returns>Enumerable collection of actions to perform after test teardown.</returns>
        public IEnumerator AfterTest(ITest test)
        {
            PerformanceTest.EndTest(test);
            yield return null;
        }
    }
}
