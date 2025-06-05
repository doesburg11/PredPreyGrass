using System;
using System.Collections;
using System.Diagnostics;
using Unity.PerformanceTesting.Data;
using Unity.PerformanceTesting.Runtime;
using Unity.PerformanceTesting.Statistics;
using UnityEngine;
using Debug = UnityEngine.Debug;
using Object = UnityEngine.Object;

namespace Unity.PerformanceTesting.Measurements
{
    /// <summary>
    /// Allows measuring of frame times.
    /// </summary>
    public class FramesMeasurement
    {
        private const int k_MinTestTimeMs = 500;
        private const int k_MinWarmupTimeMs = 80;
        private const int k_ProbingMultiplier = 4;
        internal const int k_MinIterations = 7;
        internal const int k_MaxDynamicMeasurements = 1000;
        private const double k_DefaultMaxRelativeError = 0.02;
        private const ConfidenceLevel k_DefaultConfidenceLevel = ConfidenceLevel.L99;
        private const OutlierMode k_DefaultOutlierMode = OutlierMode.Remove;

        private SampleGroup[] m_ProfilerSampleGroups;
        private SampleGroup m_SampleGroup = new SampleGroup("FrameTime");
        private int m_DesiredFrameCount;
        internal bool m_DynamicMeasurementCount;
        private double m_MaxRelativeError = k_DefaultMaxRelativeError;
        private ConfidenceLevel m_ConfidenceLevel = k_DefaultConfidenceLevel;
        private OutlierMode m_OutlierMode = k_DefaultOutlierMode;
        private int m_Executions;
        private int m_Warmup = -1;
        private bool m_RecordFrametime = true;
        
        /// <summary>
        /// Records provided profiler markers once per frame.
        /// </summary>
        /// <param name="profilerMarkerNames">Profiler marker names as in profiler window.</param>
        /// <returns>An updated instance of the FramesMeasurement to be used in fluent syntax.</returns>
        public FramesMeasurement ProfilerMarkers(params string[] profilerMarkerNames)
        {
            m_ProfilerSampleGroups = Utils.CreateSampleGroupsFromMarkerNames(profilerMarkerNames);
            return this;
        }

        /// <summary>
        /// Records provided profiler markers once per frame.
        /// </summary>
        /// <param name="sampleGroups">List of SampleGroups where a name matches the profiler marker and desired SampleUnit</param>
        /// <returns>An updated instance of the FramesMeasurement to be used in fluent syntax.</returns>
        public FramesMeasurement ProfilerMarkers(params SampleGroup[] sampleGroups)
        {
            m_ProfilerSampleGroups = sampleGroups;
            return this;
        }

        /// <summary>
        /// Overrides the name of default sample group "Time".
        /// </summary>
        /// <param name="name">Name of the sample group.</param>
        /// <returns>An updated instance of the FramesMeasurement to be used in fluent syntax.</returns>
        public FramesMeasurement SampleGroup(string name)
        {
            m_SampleGroup.Name = name; 
            return this;
        }

        /// <summary>
        /// Overrides the default sample group "Time"
        /// </summary>
        /// <param name="sampleGroup">Sample group to use.</param>
        /// <returns>An updated instance of the FramesMeasurement to be used in fluent syntax.</returns>
        public FramesMeasurement SampleGroup(SampleGroup sampleGroup)
        {
            m_SampleGroup = sampleGroup;
            return this;
        }

        /// <summary>
        /// Count of measurements to take.
        /// </summary>
        /// <param name="count">Count of measurements.</param>
        /// <returns>An updated instance of the FramesMeasurement to be used in fluent syntax.</returns>
        public FramesMeasurement MeasurementCount(int count)
        {
            m_Executions = count;
            return this;
        }

        /// <summary>
        /// Dynamically find a suitable measurement count based on the margin of error of the samples.
        /// The measurements will stop once a certain amount of samples (specified by a confidence interval)
        /// falls within an acceptable error range from the result (defined by a relative error of the mean).
        /// A default margin of error range of 2% and a default confidence interval of 99% will be used.
        /// </summary>
        /// <param name="outlierMode">Outlier mode allows to include or exclude outliers when evaluating the stop criterion.</param>
        /// <returns>An updated instance of the FramesMeasurement to be used in fluent syntax.</returns>
        public FramesMeasurement DynamicMeasurementCount(OutlierMode outlierMode = k_DefaultOutlierMode)
        {
            m_DynamicMeasurementCount = true;
            m_OutlierMode = outlierMode;
            return this;
        }

        /// <summary>
        /// Dynamically find a suitable measurement count based on the margin of error of the samples.
        /// The measurements will stop once a certain amount of samples (specified by a confidence interval)
        /// falls within an acceptable error range from the result (defined by a relative error of the mean).
        /// </summary>
        /// <param name="maxRelativeError">The maximum relative error of the mean that the margin of error must fall into.</param>
        /// <param name="confidenceLevel">The confidence interval which will be used to calculate the margin of error.</param>
        /// <param name="outlierMode">Outlier mode allows to include or exclude outliers when evaluating the stop criterion.</param>
        /// <returns>An updated instance of the FramesMeasurement to be used in fluent syntax.</returns>
        public FramesMeasurement DynamicMeasurementCount(double maxRelativeError, ConfidenceLevel confidenceLevel = k_DefaultConfidenceLevel,
            OutlierMode outlierMode = k_DefaultOutlierMode)
        {
            m_MaxRelativeError = maxRelativeError;
            m_ConfidenceLevel = confidenceLevel;
            m_DynamicMeasurementCount = true;
            m_OutlierMode = outlierMode;
            return this;
        }

        /// <summary>
        /// Count of warmup executions.
        /// </summary>
        /// <param name="count">Count of warmup executions.</param>
        /// <returns>An updated instance of the FramesMeasurement to be used in fluent syntax.</returns>
        public FramesMeasurement WarmupCount(int count)
        {
            m_Warmup = count;
            return this;
        }

        /// <summary>
        /// Specifies frame times should not be recorded.
        /// </summary>
        /// <returns>An updated instance of the FramesMeasurement to be used in fluent syntax.</returns>
        public FramesMeasurement DontRecordFrametime()
        {
            m_RecordFrametime = false;
            return this;
        }

        /// <summary>
        /// Switches frame time measurement to asynchronous scope measurement.
        /// </summary>
        /// <param name="name">Sample group name.</param>
        /// <returns>A ScopedFrameTimeMeasurement instance configured according to the given arguments.</returns>
        public ScopedFrameTimeMeasurement Scope(string name = "Time")
        {
            return new ScopedFrameTimeMeasurement(name);
        }
        
        /// <summary>
        /// Switches frame time measurement to asynchronous scope measurement.
        /// </summary>
        /// <param name="sampleGroup">Sample group to save measurements.</param>
        /// <returns>A ScopedFrameTimeMeasurement instance configured according to the given arguments.</returns>
        public ScopedFrameTimeMeasurement Scope(SampleGroup sampleGroup)
        {
            return new ScopedFrameTimeMeasurement(sampleGroup);
        }
        
        /// <summary>
        /// Executes the frame time measurement with given parameters. When MeasurementCount is not provided, a probing method will run to determine desired measurement counts.
        /// </summary>
        /// <returns>IEnumerator to yield until finish.</returns>
        public IEnumerator Run()
        {
            ValidateCorrectDynamicMeasurementCountUsage();

            if (!ValidateMeasurementAndWarmupCount()) yield break;
            
            SettingsOverride();

            yield return m_Warmup > -1 ? WaitFor(m_Warmup) : GetDesiredIterationCount();

            using (Measure.ProfilerMarkers(m_ProfilerSampleGroups))
            {
                if (m_DynamicMeasurementCount)
                {
                    yield return RunDynamicMeasurementCount();
                }
                else
                {
                    yield return RunFixedMeasurementCount();
                }

                // WaitForEndOfFrame coroutine is not invoked on the editor in batch mode
                // This may lead to unexpected behavior and is better to avoid
                // https://docs.unity3d.com/ScriptReference/WaitForEndOfFrame.html
                if (!Application.isBatchMode && Application.isPlaying)
                {
                    yield return new WaitForEndOfFrame();
                }
            }
        }
        
        private IEnumerator RunDynamicMeasurementCount()
        {
            while (true)
            {
                using (Measure.Scope(m_SampleGroup))
                {
                    yield return null;
                }
                if (SampleCountFulfillsRequirements())
                    break;
            }
        }

        private IEnumerator RunFixedMeasurementCount()
        {
            m_DesiredFrameCount = m_Executions > 0 ? m_Executions : m_DesiredFrameCount;

            for (var i = 0; i < m_DesiredFrameCount; i++)
            {
                if (m_RecordFrametime)
                {
                    using (Measure.Scope(m_SampleGroup))
                    {
                        yield return null;
                    }
                }
                else
                {
                    yield return null;
                }
            }
        }

        private bool ValidateMeasurementAndWarmupCount()
        {
            if (m_DynamicMeasurementCount || m_Executions != 0 || m_Warmup < 0) return true;
            Debug.LogError("Provide execution count or remove warmup count from frames measurement.");
            return false;

        }

        private void ValidateCorrectDynamicMeasurementCountUsage()
        {
            if (!m_DynamicMeasurementCount)
                return;

            if (m_Executions > 0)
            {
                m_DynamicMeasurementCount = false;
                Debug.LogWarning("DynamicMeasurementCount will be ignored because MeasurementCount was specified.");
                return;
            }

            if (!m_RecordFrametime)
            {
                m_DynamicMeasurementCount = false;
                Debug.LogWarning("DynamicMeasurementCount will be ignored because FrameTime measurement was disabled.");
            }
        }

        private bool SampleCountFulfillsRequirements()
        {
            var samples = m_SampleGroup.Samples;
            var sampleCount = samples.Count;
            var statistics = MeasurementsStatistics.Calculate(samples, m_OutlierMode, m_ConfidenceLevel);
            var actualError = statistics.MarginOfError;
            var maxError = m_MaxRelativeError * statistics.Mean;

            if (sampleCount >= k_MinIterations && actualError < maxError)
                return true;

            if (sampleCount >= k_MaxDynamicMeasurements)
                return true;

            return false;
        }

        /// <summary>
        /// Overrides measurement count based on performance run settings
        /// </summary>
        private void SettingsOverride()
        {
            var count = RunSettings.Instance.MeasurementCount;
            if (count < 0) { return; }
            m_Executions = count;
            m_Warmup = m_Warmup < 1 ? 0 : count;
            m_DynamicMeasurementCount = false;
        }

        private IEnumerator GetDesiredIterationCount()
        {
            var executionTime = 0.0D;
            var iterations = 1;

            while (executionTime < k_MinWarmupTimeMs)
            {
                var sw = Stopwatch.GetTimestamp();

                yield return WaitFor(iterations);

                executionTime = TimeSpan.FromTicks(Stopwatch.GetTimestamp() - sw).TotalMilliseconds;

                if (iterations == 1 && executionTime > 40)
                {
                    m_DesiredFrameCount = k_MinIterations;
                    yield break;
                }

                if (iterations == 64)
                {
                    m_DesiredFrameCount = 120;
                    yield break;
                }

                if (executionTime < k_MinWarmupTimeMs)
                {
                    iterations *= k_ProbingMultiplier;
                }
            }

            m_DesiredFrameCount = (int)(k_MinTestTimeMs * iterations / executionTime);
        }

        private IEnumerator WaitFor(int iterations)
        {
            for (var i = 0; i < iterations; i++)
            {
                yield return null;
            }
        }

        /// <summary>
        /// Provides a way to measure frame time within a scope.
        /// </summary>
        public struct ScopedFrameTimeMeasurement : IDisposable
        {
            private readonly FrameTimeMeasurement m_Test;

            /// <summary>
            /// Initializes a scoped frame time measurement.
            /// </summary>
            /// <param name="sampleGroup">Sample group used to store measurements.</param>
            public ScopedFrameTimeMeasurement(SampleGroup sampleGroup)
            {
                var go = new GameObject("Recorder");
                if (Application.isPlaying) Object.DontDestroyOnLoad(go);
                m_Test = go.AddComponent<FrameTimeMeasurement>();
                m_Test.SampleGroup = sampleGroup;
                PerformanceTest.Disposables.Add(this);
            }
            
            /// <summary>
            /// Initializes a scoped frame time measurement.
            /// </summary>
            /// <param name="name">Sample group name used to store measurements.</param>
            public ScopedFrameTimeMeasurement(string name): this(new SampleGroup(name))
            {
            }

            /// <summary>
            /// Stops scoped frame time measurement and adds it to provided sample group.
            /// </summary>
            public void Dispose()
            {
                PerformanceTest.Disposables.Remove(this);
                Object.DestroyImmediate(m_Test.gameObject);
            }
        }
    }
}
