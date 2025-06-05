using System;
using System.Diagnostics;
using Unity.PerformanceTesting.Runtime;

namespace Unity.PerformanceTesting.Measurements
{
    /// <summary>
    /// Measures execution time for the given scope as a single time.
    /// </summary>
    public struct ScopeMeasurement : IDisposable
    {
        private readonly SampleGroup m_SampleGroup;
        private readonly long m_StartTicks;

        /// <summary>
        /// Initializes a scope measurement.
        /// </summary>
        /// <param name="sampleGroup">Sample group used to save measurements.</param>
        public ScopeMeasurement(SampleGroup sampleGroup)
        {
            m_SampleGroup = PerformanceTest.GetSampleGroup(sampleGroup.Name);
            if (m_SampleGroup == null)
            {
                m_SampleGroup = sampleGroup;
                PerformanceTest.Active.SampleGroups.Add(m_SampleGroup);
            }

            m_StartTicks = Stopwatch.GetTimestamp();
            PerformanceTest.Disposables.Add(this);
        }

        /// <summary>
        /// Initializes a scope measurement.
        /// </summary>
        /// <param name="name">Sample group name used for measurements.</param>
        public ScopeMeasurement(string name) : this(new SampleGroup(name))
        {
        }

        /// <summary>
        /// Stops scope measurement and adds it to provided sample group.
        /// </summary>
        public void Dispose()
        {
            var elapsedTicks = Stopwatch.GetTimestamp() - m_StartTicks;
            PerformanceTest.Disposables.Remove(this);
            var delta = TimeSpan.FromTicks(elapsedTicks).TotalMilliseconds;
            
            delta = Utils.ConvertSample(SampleUnit.Millisecond, m_SampleGroup.Unit, delta);

            Measure.Custom(m_SampleGroup, delta);
        }
    }
}