using System;

namespace Unity.PerformanceTesting.Statistics
{
    /// <summary>
    /// Represents a statistical confidence level.
    /// </summary>
    public enum ConfidenceLevel
    {
        /// <summary>
        /// 90.0% confidence interval
        /// </summary>
        L90,

        /// <summary>
        /// 95.0% confidence interval
        /// </summary>
        L95,

        /// <summary>
        /// 99.0% confidence interval
        /// </summary>
        L99,

        /// <summary>
        /// 99.9% confidence interval
        /// </summary>
        L999
    }
}
