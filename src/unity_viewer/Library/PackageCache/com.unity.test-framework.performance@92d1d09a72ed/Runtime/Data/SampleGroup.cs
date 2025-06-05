using System;
using System.Collections.Generic;
using System.Runtime.Serialization;
using Unity.PerformanceTesting.Exceptions;

namespace Unity.PerformanceTesting.Data
{
    /// <summary>
    /// Represents a group of samples for a performance test that share a common name and unit.
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
        /// Median value of samples.
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
        /// Creates a sample group without initializing values.
        /// </summary>
        public SampleGroup(){}
    
        /// <summary>
        /// Creates a sample group with provided arguments.
        /// </summary>
        /// <param name="name">Sample group name.</param>
        /// <param name="unit">Measurement unit.</param>
        /// <param name="increaseIsBetter">Whether the measurement is inverted and increase is positive.</param>
        /// <exception cref="PerformanceTestException">Exception thrown when invalid name is used.</exception>
        public SampleGroup(string name, SampleUnit unit, bool increaseIsBetter)
        {
            Name = name;
            Unit = unit;
            IncreaseIsBetter = increaseIsBetter;

            if (string.IsNullOrEmpty(name))
            {
                throw new PerformanceTestException("Sample group name is empty. Please assign a valid name.");
            }
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
