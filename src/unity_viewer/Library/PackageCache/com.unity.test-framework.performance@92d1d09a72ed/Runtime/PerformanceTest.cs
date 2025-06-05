using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Text;
using Unity.PerformanceTesting.Runtime;
using NUnit.Framework;
using NUnit.Framework.Interfaces;
using Unity.PerformanceTesting.Exceptions;
using UnityEngine;
using UnityEngine.TestRunner.NUnitExtensions;

[assembly: InternalsVisibleTo("Unity.PerformanceTesting.Tests.Editor")]
namespace Unity.PerformanceTesting
{
    /// <summary>
    /// Represents active performance test as a singleton.
    /// </summary>
    [Serializable]
    public class PerformanceTest
    {
        /// <summary>
        /// Full name of the test.
        /// </summary>
        public string Name;
        /// <summary>
        /// Class name of the test.
        /// </summary>
        public string ClassName;
        /// <summary>
        /// Method name of the test.
        /// </summary>
        public string MethodName;
        /// <summary>
        /// Version of the test. Default "1".
        /// </summary>
        public string Version;
        /// <summary>
        /// List of categories assigned to the test.
        /// </summary>
        public List<string> Categories = new List<string>();
        /// <summary>
        /// List of sample groups assigned to the test.
        /// </summary>
        public List<SampleGroup> SampleGroups = new List<SampleGroup>();
        /// <summary>
        /// Singleton instance of active performance test.
        /// </summary>
        public static PerformanceTest Active { get; set; }
        private static List <IDisposable> m_Disposables = new List<IDisposable>(1024);
        internal static List<IDisposable> Disposables
        {
            get => m_Disposables;
            set => m_Disposables = value ?? new List<IDisposable>(1024);
        }
        PerformanceTestHelper m_PerformanceTestHelper;

        /// <summary>
        /// An action to be invoked when a performance test has finished execution.
        /// </summary>
        public static event Action OnTestEnded;

        /// <summary>
        /// Initializes a new performance test and assigns it as singleton.
        /// </summary>
        public PerformanceTest()
        {
            Active = this;
        }

        internal static void StartTest(ITest currentTest)
        {
            if (currentTest.IsSuite) return;

            var go = new GameObject("PerformanceTestHelper");
            go.hideFlags = HideFlags.HideAndDontSave;
            var performanceTestHelper = go.AddComponent<PerformanceTestHelper>();

            string methodName = currentTest.Name.Contains("(")
                ? currentTest.Name.Remove(currentTest.Name.IndexOf("(", StringComparison.Ordinal))
                : currentTest.Name;

            string className = currentTest.ClassName;
            
            var fullName = currentTest.MethodName != methodName ? $"{currentTest.ClassName}.{currentTest.MethodName}.{currentTest.Name}" : currentTest.FullName;

            var test = new PerformanceTest
            {
                Name = fullName,
                ClassName = className,
                MethodName = methodName,
                Categories = currentTest.GetAllCategoriesFromTest(),
                Version = GetVersion(currentTest),
                m_PerformanceTestHelper = performanceTestHelper
            };

            Active = test;
            performanceTestHelper.ActiveTest = test;
        }

        private static string GetVersion(ITest currentTest)
        {
            string version = "";
            var methodVersions = currentTest.Method.GetCustomAttributes<VersionAttribute>(false);
            var classVersion = currentTest.TypeInfo.Type.GetCustomAttributes(typeof(VersionAttribute), true);

            if (classVersion.Length > 0)
                version = ((VersionAttribute)classVersion[0]).Version + ".";
            if (methodVersions.Length > 0)
                version += methodVersions[0].Version;
            else
                version += "1";

            return version;
        }

        internal static void EndTest(ITest test)
        {
            if (test.IsSuite) return;

            if (Active.m_PerformanceTestHelper != null && Active.m_PerformanceTestHelper.gameObject != null)
            {
                UnityEngine.Object.DestroyImmediate(Active.m_PerformanceTestHelper.gameObject);
            }

            DisposeMeasurements();
            Active.CalculateStatisticalValues();

            try
            {
                // Notify subscribers that the test has ended by invoking OnTestEnded event
                OnTestEnded?.Invoke();
            }
            catch (Exception ex)
            {
                // An exception occurred while invoking the OnTestEnded event.
                // Log the error message, exception type, and stack trace for troubleshooting.
                Debug.LogError($"An exception occurred in OnTestEnd callback: {ex.GetType()}: {ex.Message}\n{ex.StackTrace}");
            }
            finally
            {
                // Regardless of whether the event invocation succeeded or not, perform cleanup
                // and finalize the test-related operations.
                PerformCleanupAndFinalization();
            }
        }
        
        internal static void PerformCleanupAndFinalization()
        {
            Active.LogOutput(); // Log test output
            TestContext.Out.WriteLine("##performancetestresult2:" + Active.Serialize()); // Log test result
            PlayerCallbacks.LogMetadata(); // Log metadata
            Active = null; // Clear active object
            GC.Collect(); // Trigger garbage collection to free resources
        }

        private static void DisposeMeasurements()
        {
            for (var i = 0; i < Disposables.Count; i++)
            {
                Disposables[i].Dispose();
            }

            Disposables.Clear();
        }

        /// <summary>
        /// Retrieves named sample group from active performance test.
        /// </summary>
        /// <param name="name">Name of sample group to retrieve.</param>
        /// <returns>Selected sample group.</returns>
        /// <exception cref="PerformanceTestException">Exception will be thrown if there is no active performance test.</exception>
        public static SampleGroup GetSampleGroup(string name)
        {
            if (Active == null) throw new PerformanceTestException("Trying to record samples but there is no active performance tests.");
            foreach (var sampleGroup in Active.SampleGroups)
            {
                if (sampleGroup.Name == name)
                    return sampleGroup;
            }

            return null;
        }

        /// <summary>
        /// Adds sample group to active performance test.
        /// </summary>
        /// <param name="sampleGroup">Sample group to be added.</param>
        public static void AddSampleGroup(SampleGroup sampleGroup)
        {
            Active.SampleGroups.Add(sampleGroup);
        }

        internal string Serialize()
        {
            return JsonUtility.ToJson(Active);
        }

        /// <summary>
        /// Loops through sample groups and updates statistical values.
        /// </summary>
        public void CalculateStatisticalValues()
        {
            foreach (var sampleGroup in SampleGroups)
            {
                sampleGroup.UpdateStatistics();
            }
        }

        private void LogOutput()
        {
            TestContext.Out.WriteLine(ToString());
        }

        static void AppendVisualization(StringBuilder sb, IList<double> data, int n, double min, double max)
        {
            const string bars = "▁▂▃▄▅▆▇█";
            double range = max - min;
            for (int i = 0; i < n; i++)
            {
                var sample = data[i];
                int idx = Mathf.Clamp(Mathf.RoundToInt((float) ((sample - min) / range * (bars.Length - 1))), 0, bars.Length - 1);
                sb.Append(bars[idx]);
            }
        }

        private static double[] s_Buckets;
        static void AppendSampleHistogram(StringBuilder sb, SampleGroup s, int buckets)
        {
            if (s_Buckets == null || s_Buckets.Length < buckets)
                s_Buckets = new double[buckets];
            double maxInOneBucket = 0;
            double min = s.Min;
            double bucketsOverRange = (buckets - 1) / (s.Max - s.Min);
            for (int i = 0; i < s.Samples.Count; i++)
            {
                int bucket = Mathf.Clamp(Mathf.RoundToInt((float)((s.Samples[i] - min) * bucketsOverRange)), 0, buckets - 1);
                s_Buckets[bucket] += 1;
                if (s_Buckets[bucket] > maxInOneBucket)
                    maxInOneBucket = s_Buckets[bucket];
            }
            AppendVisualization(sb, s_Buckets, s_Buckets.Length, 0, maxInOneBucket);
        }
        
        /// <summary>
        /// Returns performance test in a readable format.
        /// </summary>
        /// <returns>Readable representation of performance test.</returns>
        public override string ToString()
        {
            var logString = new StringBuilder();

            foreach (var s in SampleGroups)
            {
                logString.Append(s.Name);

                if (s.Samples.Count == 1)
                {
                    logString.AppendLine($" {s.Samples[0]:0.00} {s.Unit}s");
                }
                else
                {
                    string u = s.Unit.ShortName();
                    logString.AppendLine($" in {s.Unit}s\nMin:\t\t{s.Min:0.00} {u}\nMedian:\t\t{s.Median:0.00} {u}\nMax:\t\t{s.Max:0.00} {u}\nAvg:\t\t{s.Average:0.00} {u}\nStdDev:\t\t{s.StandardDeviation:0.00} {u}\nSampleCount:\t{s.Samples.Count}\nSum:\t\t{s.Sum:0.00} {u}");
                    logString.Append("First samples:\t");
                    AppendVisualization(logString, s.Samples, Mathf.Min(s.Samples.Count, 100), s.Min, s.Max);
                    logString.AppendLine();
                    if (s.Samples.Count <= 512)
                    {
                        int numBuckets = Mathf.Min(10, s.Samples.Count / 4);
                        if (numBuckets > 2)
                        {
                            logString.Append("Histogram:\t");
                            AppendSampleHistogram(logString, s, numBuckets);
                            logString.AppendLine();
                        }
                        else
                            logString.Append("(not enough samples for histogram)\n");
                    }
                    logString.AppendLine();
                }
            }

            return logString.ToString();
        }
    }
}
