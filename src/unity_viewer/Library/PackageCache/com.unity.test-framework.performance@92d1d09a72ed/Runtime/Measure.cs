using System;
using System.Diagnostics;
using Unity.PerformanceTesting.Exceptions;
using Unity.PerformanceTesting.Measurements;
using Unity.PerformanceTesting.Runtime;
using UnityEngine;
using Object = UnityEngine.Object;

namespace Unity.PerformanceTesting
{
    /// <summary>
    /// Enables measuring of performance metrics during a performance test.
    /// </summary>
    public static class Measure
    {
        /// <summary>
        /// Saves provided value as a performance measurement.
        /// </summary>
        /// <param name="sampleGroup">The sample group to save the value to.</param>
        /// <param name="value">Value to be saved.</param>
        public static void Custom(SampleGroup sampleGroup, double value)
        {
            VerifyValue(sampleGroup.Name, value);

            var activeSampleGroup = PerformanceTest.GetSampleGroup(sampleGroup.Name);
            if (activeSampleGroup == null)
            {
                PerformanceTest.AddSampleGroup(sampleGroup);
                activeSampleGroup = sampleGroup;
            }

            activeSampleGroup.Samples.Add(value);
        }

        /// <summary>
        /// Saves provided value as a performance measurement.
        /// </summary>
        /// <param name="name">The name of the sample group to save the value to.</param>
        /// <param name="value">Value to be saved.</param>
        public static void Custom(string name, double value)
        {
            VerifyValue(name, value);

            var activeSampleGroup = PerformanceTest.GetSampleGroup(name);
            if (activeSampleGroup == null)
            {
                activeSampleGroup = new SampleGroup(name);
                PerformanceTest.AddSampleGroup(activeSampleGroup);
            }

            activeSampleGroup.Samples.Add(value);
        }

        static void VerifyValue(string name, double value)
        {
            if (double.IsNaN(value))
                throw new PerformanceTestException(
                    $"Trying to record value which is not a number for sample group: {name}");
        }

        /// <summary>
        /// Measures execution time for the given scope as a single time.
        /// </summary>
        /// <param name="name">Name to use for the sample group.</param>
        /// <returns>IDisposable <see cref="ScopeMeasurement"/> on which you should call the <see cref="ScopeMeasurement.Dispose"/> method to stop measurement.</returns>
        public static ScopeMeasurement Scope(string name = "Time")
        {
            return new ScopeMeasurement(name);
        }
        
        /// <summary>
        /// Measures execution time for the given scope as a single time.
        /// </summary>
        /// <param name="sampleGroup">Sample group to use to save samples to.</param>
        /// <returns>IDisposable <see cref="ScopeMeasurement"/> on which you should call the <see cref="ScopeMeasurement.Dispose"/> method to stop measurement.</returns>
        public static ScopeMeasurement Scope(SampleGroup sampleGroup)
        {
            return new ScopeMeasurement(sampleGroup);
        }

        /// <summary>
        /// Measures profiler markers for the given scope.
        /// </summary>
        /// <param name="profilerMarkerLabels">List of profiler marker names.</param>
        /// <returns>A ProfilerMeasurement instance configured according to the given arguments.</returns>
        public static ProfilerMeasurement ProfilerMarkers(params string[] profilerMarkerLabels)
        {
            return new ProfilerMeasurement(profilerMarkerLabels);
        }
        
        /// <summary>
        /// Measures profiler markers for the given scope.
        /// </summary>
        /// <param name="sampleGroups">List of SampleGroups where the name matches the profiler marker to measure.</param>
        /// <returns>A ProfilerMeasurement instance configured according to the given arguments.</returns>
        public static ProfilerMeasurement ProfilerMarkers(params SampleGroup[] sampleGroups)
        {
            return new ProfilerMeasurement(sampleGroups);
        }

        /// <summary>
        /// Measures execution time for a method with given parameters.
        /// </summary>
        /// <param name="action">The action to be measured.</param>
        /// <returns><see cref="MethodMeasurement"/>using a builder pattern to provide parameters. Call <see cref="ScopeMeasurement.Run"/> to start measurement.</returns>
        public static MethodMeasurement Method(Action action)
        {
            return new MethodMeasurement(action);
        }

        /// <summary>
        /// Measures frame times with given parameters.
        /// </summary>
        /// <returns><see cref="FramesMeasurement"/> using a builder pattern to provide parameters. Call <see cref="FramesMeasurement.Run"/> method to start measurement.</returns>
        public static FramesMeasurement Frames()
        {
            return new FramesMeasurement();
        }
    }
}
