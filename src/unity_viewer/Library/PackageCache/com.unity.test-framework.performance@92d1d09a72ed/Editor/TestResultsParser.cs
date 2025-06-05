using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;
using System.Xml.Linq;
using Unity.PerformanceTesting.Data;
using UnityEditor.TestTools.TestRunner.Api;
using UnityEngine;

namespace Unity.PerformanceTesting.Editor
{
    static class TestResultsParser
    {
        internal static Run GetPerformanceTestRunData(ITestResultAdaptor testResults)
        {
            var testOutputs = GetTestOutputsRecursively(testResults);
            return ExtractPerformanceRunData(testOutputs);
        }

        internal static Run GetPerformanceTestRunDataFromXmlFile(string xmlResultsPath)
        {
            try
            {
                var xmlDocument = XDocument.Load(xmlResultsPath);
                return GetPerformanceTestRunData(xmlDocument);
            }
            catch (Exception e)
            {
                Debug.LogWarning("Failed to load performance test results from XML.\n" +
                    $"{e.GetType()}: {e.Message}\n" +
                    $"{e.StackTrace}");
            }

            return null;
        }

        static Run GetPerformanceTestRunData(XDocument testResults)
        {
            var testOutputs = testResults.Descendants("output")
                .Select(outputElement => outputElement.Value)
                .ToArray();

            return ExtractPerformanceRunData(testOutputs);
        }

        static string[] GetTestOutputsRecursively(ITestResultAdaptor testResults)
        {
            var res = new List<string>();
            if (testResults != null)
            {
                AccumulateTestRunOutputRecursively(testResults, res);
            }

            return res.ToArray();
        }

        static void AccumulateTestRunOutputRecursively(ITestResultAdaptor parentResult, List<string> outputs)
        {
            var children = parentResult.Children;
            foreach (var child in children)
            {
                AccumulateTestRunOutputRecursively(child, outputs);
            }

            var currentTestOutput = parentResult.Output;
            if (!string.IsNullOrEmpty(currentTestOutput))
            {
                outputs.Add(currentTestOutput);
            }
        }

        internal static Run ExtractPerformanceRunData(string[] testOutputs)
        {
            if (testOutputs == null || testOutputs.Length == 0)
            {
                return null;
            }

            try
            {
                var run = ExtractPerformanceTestRunInfo(testOutputs);
                if (run != null)
                {
                    DeserializeTestResults(testOutputs, run);
                }

                return run;
            }
            catch (FormatException fe)
            {
                Debug.LogError($"Invalid performance test results format: {fe.Message}");
            }
            catch (Exception e)
            {
                Debug.LogError($"Unexpected exception while reading performance test results: {e.Message}");
            }

            return null;
        }

        static Run ExtractPerformanceTestRunInfo(string[] testOutputs)
        {
            foreach (var output in testOutputs)
            {
                const string pattern = @"##performancetestruninfo2:(.+)\n";
                var regex = new Regex(pattern);
                var matches = regex.Match(output);
                if (matches.Groups.Count == 0 || matches.Captures.Count == 0)
                {
                    continue;
                }

                if (matches.Groups[1].Captures.Count > 1)
                {
                    throw new FormatException("Multiple execution metadata instances found.");
                }

                var json = matches.Groups[1].Value;
                if (string.IsNullOrEmpty(json))
                {
                    throw new FormatException("No execution metadata found.");
                }

                return ReadPerformanceTestRunJsonObject(json);
            }

            return null;
        }

        static Run ReadPerformanceTestRunJsonObject(string json)
        {
            try
            {
                return JsonUtility.FromJson<Run>(json);
            }
            catch (Exception e)
            {
                throw new FormatException($"Failed to read performance execution metadata from json string: '{json}'.\n" +
                    $"Exception: {e.Message}\n" +
                    $"{e.StackTrace}");
            }
        }

        static void DeserializeTestResults(string[] testOutputs, Run run)
        {
            foreach (var output in testOutputs)
            {
                foreach (var line in output.Split('\n'))
                {
                    var json = GetJsonFromHashtag("performancetestresult2", line);
                    if (json == null)
                    {
                        continue;
                    }

                    var result = ReadPerformanceTestResultJsonObject(json);
                    if (result != null)
                    {
                        run.Results.Add(result);
                    }
                }
            }
        }

        static PerformanceTestResult ReadPerformanceTestResultJsonObject(string json)
        {
            try
            {
                return JsonUtility.FromJson<PerformanceTestResult>(json);
            }
            catch (Exception e)
            {
                Debug.LogWarning($"Failed to read performance results from json string: '{json}'.\n" +
                    $"Exception: {e.Message}\n" +
                    $"{e.StackTrace}");
            }

            return null;
        }

        static string GetJsonFromHashtag(string tag, string line)
        {
            if (!line.Contains($"##{tag}:"))
            {
                return null;
            }

            var jsonStart = line.IndexOf('{');
            var openBrackets = 0;
            var stringIndex = jsonStart;
            while (openBrackets > 0 || stringIndex == jsonStart)
            {
                var character = line[stringIndex];
                switch (character)
                {
                    case '{':
                        openBrackets++;
                        break;
                    case '}':
                        openBrackets--;
                        break;
                }

                stringIndex++;
            }

            var jsonEnd = stringIndex;
            return line.Substring(jsonStart, jsonEnd - jsonStart);
        }
    }
}
