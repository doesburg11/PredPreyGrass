using System;
using System.Collections.Generic;
using System.Runtime.Serialization;
using Unity.PerformanceTesting.Exceptions;
using UnityEngine.Profiling;

namespace Unity.PerformanceTesting
{
    /// <summary>
    /// Represents a performance test sample group.
    /// </summary>
    [Serializable]
    public class SampleGroup : IDeserializationCallback
    {
        /// <summary>
        /// Name of the sample group.
        /// </summary>
        public string Name;
        /// <summary>
        /// Measurement unit.
        /// </summary>
        public SampleUnit Unit;
        /// <summary>
        /// Whether the measurement is inverted and increase is positive.
        /// </summary>
        public bool IncreaseIsBetter;
        /// <summary>
        /// List of samples.
        /// </summary>
        public List<double> Samples = new List<double>();
        /// <summary>
        /// Minimum value of samples.
        /// </summary>
        public double Min;
        /// <summary>
        /// Maximum value of samples.
        /// </summary>
        public double Max;
        /// <summary>
        /// Medina value of samples.
        /// </summary>
        public double Median;
        /// <summary>
        /// Average value of samples.
        /// </summary>
        public double Average;
        /// <summary>
        /// Standard deviation of samples.
        /// </summary>
        public double StandardDeviation;
        /// <summary>
        /// Sum of samples.
        /// </summary>
        public double Sum;

        /// <summary>
        /// Creates a new sample group with given parameters.
        /// </summary>
        /// <param name="name">Name of the sample group.</param>
        /// <param name="unit">Unit of measurement.</param>
        /// <param name="increaseIsBetter">Whether the measurement is inverted and increase is positive.</param>
        /// <exception cref="PerformanceTestException">Exception can be thrown if empty or null name is provided.</exception>
        public SampleGroup(string name, SampleUnit unit = SampleUnit.Millisecond, bool increaseIsBetter = false)
        {
            Name = name;
            Unit = unit;
            IncreaseIsBetter = increaseIsBetter;

            if (string.IsNullOrEmpty(name))
            {
                throw new PerformanceTestException("Sample group name is empty. Please assign a valid name.");
            }
        }
        
        internal Recorder Recorder;

        /// <summary>
        /// Gets the profiler recorder object.
        /// </summary>
        /// <returns>Profiler recorder.</returns>
        public Recorder GetRecorder()
        {
            return Recorder ?? (Recorder = Recorder.Get(Name));
        }
        
        /// <summary>	
        /// Validates the deserialized object.	
        /// </summary>	
        /// <param name="sender">The object that initiated the deserialization process.</param>	
        public void OnDeserialization(object sender)	
        {	
            if (string.IsNullOrEmpty(Name))	
            {	
                throw new PerformanceTestException("Sample group name is empty. Please assign a valid name.");	
            }	
        }
    }
}
