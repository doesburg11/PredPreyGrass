using System;
using System.Collections.Generic;
using System.Linq;

namespace Unity.PerformanceTesting.Statistics
{
    static class ConfidenceLevelExtensions
    {
        private static readonly Dictionary<ConfidenceLevel, (int value, int digits)> k_ConfidenceLevelDetails = CreateConfidenceLevelMapping();

        /// <summary>
        /// Calculates Z value (z-star) for confidence interval
        /// </summary>
        /// <param name="level">ConfidenceLevel for a confidence interval</param>
        /// <param name="n">Sample size (n >= 3)</param>
        public static double GetZValue(this ConfidenceLevel level, int n)
        {
            if (n <= 1)
                throw new ArgumentOutOfRangeException(nameof(n), "n should be >= 2");
            return StudentDistributionHelper.InverseTwoTailedStudent(1 - level.ToPercent(), n - 1);
        }

        static double ToPercent(this ConfidenceLevel level)
        {
            (int value, int digits) = k_ConfidenceLevelDetails[level];

            return value / Math.Pow(10, digits);
        }

        static Dictionary<ConfidenceLevel, (int value, int length)> CreateConfidenceLevelMapping()
        {
            return Enum.GetValues(typeof(ConfidenceLevel))
                .Cast<ConfidenceLevel>()
                .ToDictionary(
                    confidenceLevel => confidenceLevel,
                    confidenceLevel =>
                    {
                        var textRepresentation = confidenceLevel.ToString().Substring(1);
                        return (int.Parse(textRepresentation), textRepresentation.Length);
                    });
        }
    }
}
