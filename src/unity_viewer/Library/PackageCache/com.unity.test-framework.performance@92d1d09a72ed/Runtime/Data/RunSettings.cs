using System;
using System.Runtime.Serialization;
using Unity.PerformanceTesting.Runtime;

namespace Unity.PerformanceTesting.Data
{
    /// <summary>
    /// Settings store
    /// @TODO make it public along with guidelines of how and when to use
    /// @TODO add a property bag
    /// </summary>
    [Serializable]
    internal class RunSettings : IDeserializationCallback
    {
        public RunSettings(params string[] args)
        {
            if (int.TryParse(Utils.GetArg(args, "-performance-measurement-count"), out var measurementMultiplier))
            {
                MeasurementCount = measurementMultiplier;
            }
        }
        
        private static RunSettings m_Instance;

        /// <summary>
        /// Singleton instance of settings.
        /// </summary>
        public static RunSettings Instance
        {
            get
            {
                if (m_Instance == null)
                {
                    m_Instance = ResourcesLoader.Load<RunSettings>(Utils.RunSettings, Utils.PlayerPrefKeySettingsJSON);
                }
                
                return m_Instance;
            }
            set { m_Instance = value; }
        }

        /// <summary>
        /// Measurement counts will be overriden by specified value when using Measure.Method and Measure.Frames.
        /// </summary>
        public int MeasurementCount = -1;
        
        /// <summary>	
        /// Validates the deserialized object.	
        /// </summary>	
        /// <param name="sender">The object that initiated the deserialization process.</param>	
        public void OnDeserialization(object sender)	
        {	
            if (MeasurementCount < -1)	
            {	
                throw new SerializationException("MeasurementCount cannot be negative, except for the initial value of -1.");	
            }	
        }
    }
}
