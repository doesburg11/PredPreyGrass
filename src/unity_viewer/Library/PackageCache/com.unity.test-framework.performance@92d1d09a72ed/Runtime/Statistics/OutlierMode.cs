using System;

namespace Unity.PerformanceTesting.Statistics
{
    /// <summary>
    /// Outlier removal mode.
    /// </summary>
    public enum OutlierMode
    {
        /// <summary>
        /// Do not remove outliers.
        /// </summary>
        DontRemove,
            
        /// <summary>
        /// Remove outliers using the Tukey fences method.
        /// </summary>
        Remove
    }
}
