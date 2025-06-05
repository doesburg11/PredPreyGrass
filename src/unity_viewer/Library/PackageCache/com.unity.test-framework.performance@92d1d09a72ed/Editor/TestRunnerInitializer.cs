using System;
using UnityEditor;
using UnityEditor.TestRunner.CommandLineParser;
using UnityEditor.TestTools.TestRunner.Api;
using UnityEngine;

namespace Unity.PerformanceTesting.Editor
{
    [InitializeOnLoad]
    class TestRunnerInitializer
    {
        static TestRunnerInitializer()
        {
            var args = GetCmdLineArguments();
            var resultsHandler = args.IsCmdLineRun && !string.IsNullOrEmpty(args.PerfTestResults) 
                ? CreateCmdLineResultsHandler(args) 
                : ScriptableObject.CreateInstance<PerformanceTestRunSaver>();

            var api = ScriptableObject.CreateInstance<TestRunnerApi>();
            api.RegisterCallbacks(resultsHandler);
        }

        static ICallbacks CreateCmdLineResultsHandler(CmdLineArguments cmdLineArguments)
        {
            var callbacks = ScriptableObject.CreateInstance<CmdLineResultsSavingCallbacks>();
            callbacks.SetResultsLocation(cmdLineArguments.PerfTestResults);
            return callbacks;
        }

        static CmdLineArguments GetCmdLineArguments()
        {
            var isCmdLineTestRun = false;
            string resultFilePath = null;
            
            var optionSet = new CommandLineOptionSet(
                new CommandLineOption("runTests", () => { isCmdLineTestRun = true; }),
                new CommandLineOption("runEditorTests", () => { isCmdLineTestRun = true; }),
                new CommandLineOption("perfTestResults", filePath => { resultFilePath = filePath; })
            );
            optionSet.Parse(Environment.GetCommandLineArgs());

            return new CmdLineArguments(isCmdLineTestRun, resultFilePath);
        }

        class CmdLineArguments
        {
            public bool IsCmdLineRun;
            public string PerfTestResults;

            public CmdLineArguments(bool isCmdLineRun, string perfTestResults)
            {
                IsCmdLineRun = isCmdLineRun;
                PerfTestResults = perfTestResults;
            }
        }
    }
}
