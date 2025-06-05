using System;
using System.IO;
using UnityEditor.TestTools.TestRunner.Api;
using UnityEngine;

namespace Unity.PerformanceTesting.Editor
{
    [Serializable]
    class CmdLineResultsSavingCallbacks : ScriptableObject, ICallbacks
    {
        [SerializeField]
        string resultsLocation;

        void ICallbacks.RunStarted(ITestAdaptor testsToRun)
        {
            PerformanceTest.Active = null;
        }

        void ICallbacks.RunFinished(ITestResultAdaptor result)
        {
            PlayerCallbacks.Saved = false;

            try
            {
                var performanceTestRun = TestResultsParser.GetPerformanceTestRunData(result);
                if (performanceTestRun == null)
                {
                    return;
                }
                
                Debug.LogFormat(LogType.Log, LogOption.NoStacktrace, null, "Saving performance results to: {0}", resultsLocation);
                var jsonContents = JsonUtility.ToJson(performanceTestRun, true);
                CreateDirectoryIfNecessary(resultsLocation);
                File.WriteAllText(resultsLocation, jsonContents);
            }
            catch (Exception e)
            {
                Debug.LogError("Saving performance results file failed.");
                Debug.LogException(e);
            }
        }

        static void CreateDirectoryIfNecessary(string filePath)
        {
            var directoryPath = Path.GetDirectoryName(filePath);
            if (!string.IsNullOrEmpty(directoryPath) && !Directory.Exists(directoryPath))
            {
                Directory.CreateDirectory(directoryPath);
            }
        }

        void ICallbacks.TestStarted(ITestAdaptor test) { }

        void ICallbacks.TestFinished(ITestResultAdaptor result) { }

        public void SetResultsLocation(string perfTestResultsPath)
        {
            resultsLocation = perfTestResultsPath;
        }
    }
}
