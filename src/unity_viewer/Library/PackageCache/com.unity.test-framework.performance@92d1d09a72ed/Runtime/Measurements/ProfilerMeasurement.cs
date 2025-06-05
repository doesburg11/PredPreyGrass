using System;
using Unity.PerformanceTesting.Runtime;
using UnityEngine;
using Object = UnityEngine.Object;

namespace Unity.PerformanceTesting.Measurements
{
    /// <summary>
    /// Provides a way to measure profiler markers.
    /// </summary>
    public struct ProfilerMeasurement : IDisposable
    {
        private readonly ProfilerMarkerMeasurement m_Test;

        /// <summary>
        /// Initializes a profiler marker measurement.
        /// </summary>
        /// <param name="sampleGroups">List of sample groups with name set to match profiler markers to be measured.</param>
        public ProfilerMeasurement(SampleGroup[] sampleGroups)
        {
            if (sampleGroups == null)
            {
                m_Test = null;
                return;
            }

            if (sampleGroups.Length == 0)
            {
                m_Test = null;
                return;
            }

            var go = new GameObject("Recorder");
            if (Application.isPlaying) Object.DontDestroyOnLoad(go);
            go.hideFlags = HideFlags.HideAndDontSave;
            m_Test = go.AddComponent<ProfilerMarkerMeasurement>();
            m_Test.AddProfilerSampleGroup(sampleGroups);
            PerformanceTest.Disposables.Add(this);
        }

        /// <summary>
        /// Initializes a profiler marker measurement.
        /// </summary>
        /// <param name="profilerMarkers">List of profiler markers to be measured.</param>
        public ProfilerMeasurement(string[] profilerMarkers): this(Utils.CreateSampleGroupsFromMarkerNames(profilerMarkers))
        {
        }

        /// <summary>
        /// Stops profiler marker measurement and adds them to the provided sample group.
        /// </summary>
        public void Dispose()
        {
            PerformanceTest.Disposables.Remove(this);
            if (m_Test == null) return;
            m_Test.StopAndSampleRecorders();
            Object.DestroyImmediate(m_Test.gameObject);
        }
    }
}