using System;
using System.Collections.Generic;
using Unity.PerformanceTesting.Data;
using Unity.PerformanceTesting.Runtime;
using Unity.PerformanceTesting.Exceptions;
using Unity.PerformanceTesting.Meters;
using Unity.PerformanceTesting.Statistics;
using UnityEngine;
using UnityEngine.Profiling;

namespace Unity.PerformanceTesting.Measurements
{
    /// <summary>
    /// Used as a helper class to sample execution time of methods. Uses fluent pattern to build and needs to be executed with Run method.
    /// </summary>
    public class MethodMeasurement
    {
        internal const int k_MeasurementCount = 9;
        private const int k_MinMeasurementTimeMs = 100;
        private const int k_MinWarmupTimeMs = 100;
        private const int k_ProbingMultiplier = 4;
        private const int k_MaxIterations = 10000;
        internal const int k_MaxDynamicMeasurements = 1000;
        private const double k_DefaultMaxRelativeError = 0.02;
        private const ConfidenceLevel k_DefaultConfidenceLevel = ConfidenceLevel.L99;
        private const OutlierMode k_DefaultOutlierMode = OutlierMode.Remove;
        private readonly Action m_Action;
        private readonly List<SampleGroup> m_SampleGroups = new List<SampleGroup>();
        private readonly Recorder m_GCRecorder;

        private Action m_Setup;
        private Action m_Cleanup;
        private SampleGroup m_SampleGroup = new SampleGroup("Time", SampleUnit.Millisecond, false);
        private SampleGroup m_SampleGroupGC = new SampleGroup("Time.GC()", SampleUnit.Undefined, false);
        private int m_WarmupCount;
        private int m_MeasurementCount;
        internal bool m_DynamicMeasurementCount;
        private double m_MaxRelativeError = k_DefaultMaxRelativeError;
        private ConfidenceLevel m_ConfidenceLevel = k_DefaultConfidenceLevel;
        private OutlierMode m_OutlierMode = k_DefaultOutlierMode;
        private int m_IterationCount = 1;
        private bool m_GC;
        private double m_GCAccumulation;
        private IStopWatch m_Watch;

        /// <summary>
        /// Initializes a method measurement.
        /// </summary>
        /// <param name="action">Method to be measured.</param>
        public MethodMeasurement(Action action)
        {
            m_Action = action;
            m_GCRecorder = Recorder.Get("GC.Alloc");
            m_GCRecorder.enabled = false;
            if (m_Watch == null) m_Watch = new StopWatch();
        }

        internal MethodMeasurement StopWatch(IStopWatch watch)
        {
            m_Watch = watch;

            return this;
        }

        /// <summary>
        /// Will record provided profiler markers once per frame.
        /// </summary>
        /// <param name="profilerMarkerNames">Profiler marker names as in profiler window.</param>
        /// <returns>An updated instance of the MethodMeasurement to be used in fluent syntax.</returns>
        public MethodMeasurement ProfilerMarkers(params string[] profilerMarkerNames)
        {
            if (profilerMarkerNames == null) return this;
            foreach (var marker in profilerMarkerNames)
            {
                var sampleGroup = new SampleGroup(marker, SampleUnit.Nanosecond, false);
                sampleGroup.GetRecorder();
                sampleGroup.Recorder.enabled = false;
                m_SampleGroups.Add(sampleGroup);
            }

            return this;
        }

        /// <summary>
        /// Will record provided profiler markers once per frame with additional control over the SampleUnit.
        /// </summary>
        /// <param name="sampleGroups">List of SampleGroups where a name matches the profiler marker and desired SampleUnit.</param>
        /// <returns>An updated instance of the MethodMeasurement to be used in fluent syntax.</returns>
        public MethodMeasurement ProfilerMarkers(params SampleGroup[] sampleGroups)
        {
            if (sampleGroups == null){ return this;}
            foreach (var sampleGroup in sampleGroups)
            {
                sampleGroup.GetRecorder();
                sampleGroup.Recorder.enabled = false;
                m_SampleGroups.Add(sampleGroup);
            }

            return this;
        }

        /// <summary>
        /// Overrides the default SampleGroup of "Time".
        /// </summary>
        /// <param name="name">Desired name for measurement SampleGroup.</param>
        /// <returns>An updated instance of the MethodMeasurement to be used in fluent syntax.</returns>
        public MethodMeasurement SampleGroup(string name)
        {
            m_SampleGroup = new SampleGroup(name, SampleUnit.Millisecond, false);
            m_SampleGroupGC = new SampleGroup(name + ".GC()", SampleUnit.Undefined, false);
            return this;
        }
        
        /// <summary>
        /// Overrides the default SampleGroup.
        /// </summary>
        /// <param name="sampleGroup">SampleGroup with your desired name and unit.</param>
        /// <returns>An updated instance of the MethodMeasurement to be used in fluent syntax.</returns>
        public MethodMeasurement SampleGroup(SampleGroup sampleGroup)
        {
            m_SampleGroup = sampleGroup;
            m_SampleGroupGC = new SampleGroup(sampleGroup.Name + ".GC()", SampleUnit.Undefined, false);
            return this;
        }

        /// <summary>
        /// Count of times to execute before measurements are collected. If unspecified, a default warmup will be assigned.
        /// </summary>
        /// <param name="count">Count of warmup iterations to execute.</param>
        /// <returns>An updated instance of the MethodMeasurement to be used in fluent syntax.</returns>
        public MethodMeasurement WarmupCount(int count)
        {
            m_WarmupCount = count;
            return this;
        }

        /// <summary>
        /// Specifies the amount of method executions for a single measurement.
        /// </summary>
        /// <param name="count">Count of method executions.</param>
        /// <returns>An updated instance of the MethodMeasurement to be used in fluent syntax.</returns>
        public MethodMeasurement IterationsPerMeasurement(int count)
        {
            m_IterationCount = count;
            return this;
        }

        /// <summary>
        /// Specifies the number of measurements to take.
        /// </summary>
        /// <param name="count">Count of measurements to take.</param>
        /// <returns>An updated instance of the MethodMeasurement to be used in fluent syntax.</returns>
        public MethodMeasurement MeasurementCount(int count)
        {
            m_MeasurementCount = count;
            return this;
        }

        /// <summary>
        /// Dynamically find a suitable measurement count based on the margin of error of the samples.
        /// The measurements will stop once a certain amount of samples (specified by a confidence interval)
        /// falls within an acceptable error range from the result (defined by a relative error of the mean).
        /// A default margin of error range of 2% and a default confidence interval of 99% will be used.
        /// </summary>
        /// <param name="outlierMode">Outlier mode allows to include or exclude outliers when evaluating the stop criterion.</param>
        /// <returns>An updated instance of the MethodMeasurement to be used in fluent syntax.</returns>
        public MethodMeasurement DynamicMeasurementCount(OutlierMode outlierMode = k_DefaultOutlierMode)
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
        /// <returns>An updated instance of the MethodMeasurement to be used in fluent syntax.</returns>
        public MethodMeasurement DynamicMeasurementCount(double maxRelativeError, ConfidenceLevel confidenceLevel = k_DefaultConfidenceLevel,
            OutlierMode outlierMode = k_DefaultOutlierMode)
        {
            m_MaxRelativeError = maxRelativeError;
            m_ConfidenceLevel = confidenceLevel;
            m_DynamicMeasurementCount = true;
            m_OutlierMode = outlierMode;
            return this;
        }

        /// <summary>
        /// Used to provide a cleanup method which will not be measured.
        /// </summary>
        /// <param name="action">Cleanup method to execute.</param>
        /// <returns>An updated instance of the MethodMeasurement to be used in fluent syntax.</returns>
        public MethodMeasurement CleanUp(Action action)
        {
            m_Cleanup = action;
            return this;
        }

        /// <summary>
        /// Used to provide a setup method which will run before the measurement.
        /// </summary>
        /// <param name="action">Setup method to execute.</param>
        /// <returns>An updated instance of the MethodMeasurement to be used in fluent syntax.</returns>
        public MethodMeasurement SetUp(Action action)
        {
            m_Setup = action;
            return this;
        }

        /// <summary>
        /// Enables recording of garbage collector calls.
        /// </summary>
        /// <returns>An updated instance of the MethodMeasurement to be used in fluent syntax.</returns>
        public MethodMeasurement GC()
        {
            m_GC = true;
            return this;
        }

        /// <summary>
        /// Executes the measurement with given parameters. When MeasurementCount is not provided, a probing method will run to determine desired measurement counts.
        /// </summary>
        public void Run()
        {
            ValidateCorrectDynamicMeasurementCountUsage();
            SettingsOverride();
            var settingsCount = RunSettings.Instance.MeasurementCount;
            
            if (m_MeasurementCount > 0 || settingsCount > -1)
            {
                Warmup(m_WarmupCount);
                RunForIterations(m_IterationCount, m_MeasurementCount, useAverage: false);
                return;
            }

            if (m_DynamicMeasurementCount)
            {
                Warmup(m_WarmupCount);
                RunForIterations(m_IterationCount);
                return;
            }

            var iterations = Probing();
            RunForIterations(iterations, k_MeasurementCount, useAverage: true);
        }

        private void ValidateCorrectDynamicMeasurementCountUsage()
        {
            if (!m_DynamicMeasurementCount)
                return;

            if (m_MeasurementCount > 0)
            {
                m_DynamicMeasurementCount = false;
                Debug.LogWarning("DynamicMeasurementCount will be ignored because MeasurementCount was specified.");
            }
        }
        
        /// <summary>
        /// Overrides measurement count based on performance run settings
        /// </summary>
        private void SettingsOverride()
        {
            var count = RunSettings.Instance.MeasurementCount;
            if (count < 0) { return; }
            m_MeasurementCount = count;
            m_WarmupCount = m_WarmupCount > 0 ? count : 0;
            m_DynamicMeasurementCount = false;
        }

        private void RunForIterations(int iterations, int measurements, bool useAverage)
        {
            EnableMarkers();
            for (var j = 0; j < measurements; j++)
            {
                var executionTime = iterations == 1 ? ExecuteSingleIteration() : ExecuteForIterations(iterations);
                if (useAverage) executionTime /= iterations;
                var delta = Utils.ConvertSample(SampleUnit.Millisecond, m_SampleGroup.Unit, executionTime);
                Measure.Custom(m_SampleGroup, delta);
            }

            DisableAndMeasureMarkers();
        }

        private void RunForIterations(int iterations)
        {
            EnableMarkers();

            while(true)
            {
                var executionTime = iterations == 1 ? ExecuteSingleIteration() : ExecuteForIterations(iterations);
                var delta = Utils.ConvertSample(SampleUnit.Millisecond, m_SampleGroup.Unit, executionTime);
                Measure.Custom(m_SampleGroup, delta);

                if (SampleCountFulfillsRequirements())
                    break;
            }

            DisableAndMeasureMarkers();
        }

        private void EnableMarkers()
        {
            foreach (var sampleGroup in m_SampleGroups)
            {
                sampleGroup.Recorder.enabled = true;
            }
        }

        private void DisableAndMeasureMarkers()
        {
            foreach (var sampleGroup in m_SampleGroups)
            {
                sampleGroup.Recorder.enabled = false;
                var sample = sampleGroup.Recorder.elapsedNanoseconds;
                var blockCount = sampleGroup.Recorder.sampleBlockCount;
                if(blockCount == 0) continue;
                var delta = Utils.ConvertSample(SampleUnit.Nanosecond, sampleGroup.Unit, sample);
                Measure.Custom(sampleGroup, delta / blockCount);
            }
        }

        private bool SampleCountFulfillsRequirements()
        {
            var samples = m_SampleGroup.Samples;
            var sampleCount = samples.Count;
            var statistics = MeasurementsStatistics.Calculate(samples, m_OutlierMode, m_ConfidenceLevel);
            var actualError = statistics.MarginOfError;
            var maxError = m_MaxRelativeError * statistics.Mean;

            if (sampleCount >= k_MeasurementCount && actualError < maxError)
                return true;

            if (sampleCount >= k_MaxDynamicMeasurements)
                return true;

            return false;
        }

        private int Probing()
        {
            var priorGC = m_GC;
            m_GC = false;

            var executionTime = 0.0D;
            var iterations = 1;

            if (m_WarmupCount > 0)
                throw new PerformanceTestException(
                    "Please provide MeasurementCount or remove WarmupCount in your usage of Measure.Method");

            while (executionTime < k_MinWarmupTimeMs)
            {
                executionTime = m_Watch.Split();
                Warmup(iterations);
                executionTime = m_Watch.Split() - executionTime;

                if (executionTime < k_MinWarmupTimeMs)
                {
                    iterations *= k_ProbingMultiplier;
                }
            }

            if (iterations == 1)
            {
                ExecuteActionWithCleanupSetup();
                ExecuteActionWithCleanupSetup();

                m_GC = priorGC;
                return 1;
            }

            var deisredIterationsCount =
                Mathf.Clamp((int) (k_MinMeasurementTimeMs * iterations / executionTime), 1, k_MaxIterations);

            m_GC = priorGC;
            return deisredIterationsCount;
        }

        private void Warmup(int iterations)
        {
            var priorGC = m_GC;
            m_GC = false;

            for (var i = 0; i < iterations; i++)
            {
                ExecuteForIterations(m_IterationCount);
            }

            m_GC = priorGC;
        }

        private double ExecuteActionWithCleanupSetup()
        {
            m_Setup?.Invoke();
            if (m_GC) StartGCRecorder();

            var executionTime = m_Watch.Split();
            m_Action.Invoke();
            executionTime = m_Watch.Split() - executionTime;

            if (m_GC) AccumulateGCRecorder();
            m_Cleanup?.Invoke();

            return executionTime;
        }

        private double ExecuteSingleIteration()
        {
            m_Setup?.Invoke();
            if (m_GC) StartGCRecorder();

            var executionTime = m_Watch.Split();
            m_Action.Invoke();
            executionTime = m_Watch.Split() - executionTime;

            if (m_GC) EndGCRecorderAndMeasure(1);
            m_Cleanup?.Invoke();

            return executionTime;
        }

        private double ExecuteForIterations(int iterations)
        {
            var executionTime = 0.0D;

            if (m_Cleanup != null || m_Setup != null)
            {
                for (var i = 0; i < iterations; i++)
                {
                    executionTime += ExecuteActionWithCleanupSetup();
                }

                if (m_GC) EndAccumulatedGCRecorder(iterations);
            }
            else
            {
                if (m_GC) StartGCRecorder();

                executionTime = m_Watch.Split();
                for (var i = 0; i < iterations; i++)
                {
                    m_Action.Invoke();
                }

                executionTime = m_Watch.Split() - executionTime;

                if (m_GC) EndGCRecorderAndMeasure(iterations);
            }

            return executionTime;
        }

        private void StartGCRecorder()
        {
            System.GC.Collect();

            m_GCRecorder.enabled = false;
            m_GCRecorder.enabled = true;
        }

        private void AccumulateGCRecorder()
        {
            m_GCRecorder.enabled = false;
            m_GCAccumulation += m_GCRecorder.sampleBlockCount;
        }

        private void EndAccumulatedGCRecorder(int iterations)
        {
            m_GCRecorder.enabled = false;
            Measure.Custom(m_SampleGroupGC, m_GCAccumulation / iterations);
            m_GCAccumulation = 0;
        }

        private void EndGCRecorderAndMeasure(int iterations)
        {
            m_GCRecorder.enabled = false;
            Measure.Custom(m_SampleGroupGC, (double) m_GCRecorder.sampleBlockCount / iterations);
        }
    }
}
