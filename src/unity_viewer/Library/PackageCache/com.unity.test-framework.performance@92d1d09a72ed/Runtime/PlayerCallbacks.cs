using UnityEngine.TestRunner;
using Unity.PerformanceTesting;
using NUnit.Framework;
using NUnit.Framework.Interfaces;
using Unity.PerformanceTesting.Data;
using Unity.PerformanceTesting.Runtime;
using UnityEngine.Scripting;
using UnityEngine;

[assembly: TestRunCallback(typeof(PlayerCallbacks))]

namespace Unity.PerformanceTesting
{
    [Preserve]
    internal class PlayerCallbacks : ITestRunCallback
    {
        internal static bool Saved { get; set; }

        public void RunStarted(ITest testsToRun)
        {
            // This method is empty because it's part of the NUnit framework's ITestListener interface,
            // which Unity uses for running tests in the Editor. It receives a parameter "testsToRun" but
            // doesn't require implementation as Unity can execute tests without it. Developers can add 
            // custom initialization logic if needed.
        }

        public void RunFinished(ITestResult testResults)
        {
            Saved = false;
        }

        public void TestStarted(ITest test)
        {
            // This method is called by Unity when a new test has started. It receives a parameter "test"
            // which contains information about the test being executed. Developers can add custom logic
            // in this method, such as logging or setup code for the test.
        }

        public void TestFinished(ITestResult result)
        {
            // This method is called by Unity when a test has finished executing. It receives a parameter 
            // "result" which contains information about the test execution, such as whether the test 
            // passed or failed, and any messages or exceptions thrown during the test. Developers can 
            // add custom logic in this method, such as logging or teardown code for the test.
        }

        internal static void LogMetadata()
        {
            if (Saved) return;

            var run = Metadata.GetFromResources();
            var json = JsonUtility.ToJson(run);
            TestContext.Out?.WriteLine("##performancetestruninfo2:" + json);
            Saved = true;
        }
    }
}