using UnityEditor;
using UnityEditor.TestTools.TestRunner.Api;
using UnityEngine;

namespace Tests_17s
{
    public static class RunSceneValidation
    {
        [MenuItem("Tests/RunAllSceneValidations")]
        public static void ValidateScene()
        {
            var testRunnerApi = ScriptableObject.CreateInstance<TestRunnerApi>();

            testRunnerApi.RegisterCallbacks(new Callbacks());
        
            testRunnerApi.Execute(new ExecutionSettings(new Filter()
            {
                categoryNames = new[] {"SceneValidation"},
                testMode = TestMode.EditMode
            }));
        }

        private class Callbacks : ICallbacks
        {
            public void RunStarted(ITestAdaptor testsToRun)
            {
                Debug.Log("Running all scene validations...");
            }

            public void RunFinished(ITestResultAdaptor result)
            {
                Debug.Log($"Done with scene validation. Overall result {result.TestStatus}");
            }

            public void TestStarted(ITestAdaptor test)
            {
            }

            public void TestFinished(ITestResultAdaptor result)
            {
                if (result.TestStatus == TestStatus.Failed)
                {
                    Debug.Log($"Failed {result.Name}: {result.Message}");
                }
            }
        }
    }
}
