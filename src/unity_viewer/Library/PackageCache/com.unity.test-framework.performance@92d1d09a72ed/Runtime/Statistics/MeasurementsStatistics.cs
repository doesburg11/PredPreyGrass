using System;
using System.Collections.Generic;

namespace Unity.PerformanceTesting.Statistics
{
    /// <summary>
    /// Provides basic measurement statistics calculated for a sample collection.
    /// </summary>
    readonly ref struct MeasurementsStatistics
    {
        /// <summary>
        /// Mean of the samples.
        /// </summary>
        public double Mean { get; }

        /// <summary>
        /// Margin of error within the requested confidence interval.
        /// </summary>
        public double MarginOfError { get; }

        MeasurementsStatistics(double mean, double marginOfError)
        {
            Mean = mean;
            MarginOfError = marginOfError;
        }

        /// <summary>
        /// Calculates basic measurement statistics for a sample collection.
        /// </summary>
        /// <param name="measurements">The sample collection.</param>
        /// <param name="outlierMode">Outlier removal mode.</param>
        /// <param name="confidenceLevel">Confidence interval for calculating the margin of error.</param>
        /// <returns>A MeasurementStatistics instance calculated for the provided data.</returns>
        /// <exception cref="InvalidOperationException">Thrown if the sample count is zero.</exception>
        public static MeasurementsStatistics Calculate(List<double> measurements, OutlierMode outlierMode, ConfidenceLevel confidenceLevel)
        {
            var n = measurements.Count;
            if (n == 0)
                throw new InvalidOperationException("Requesting statistic measurements of a sequence which contains no elements.");

            double sum;
            double mean;
            double variance;
            double standardDeviation;
            double standardError;
            double marginOfError;

            if (outlierMode == OutlierMode.DontRemove)
            {
                sum = Sum(measurements);
                mean = sum / n;
                variance = Variance(measurements, n, mean);
                standardDeviation = Math.Sqrt(variance);
                standardError = standardDeviation / Math.Sqrt(n);
                marginOfError = n <= 2 ? double.NaN : standardError * confidenceLevel.GetZValue(n);

                return new MeasurementsStatistics(mean, marginOfError);
            }

            measurements.Sort();

            double q1, q3;
            if (n == 1)
                q1 = q3 = measurements[0];
            else
            {
                q1 = GetQuartile(measurements, measurements.Count / 2);
                q3 = GetQuartile(measurements, measurements.Count * 3 / 2);
            }

            var interquartileRange = q3 - q1;
            var lowerFence = q1 - 1.5 * interquartileRange;
            var upperFence = q3 + 1.5 * interquartileRange;

            SumWithoutOutliers(outlierMode, measurements, lowerFence, upperFence, out sum, out n);
            mean = sum / n;
            variance = VarianceWithoutOutliers(outlierMode, measurements, n, mean, lowerFence, upperFence);
            standardDeviation = Math.Sqrt(variance);
            standardError = standardDeviation / Math.Sqrt(n);
            marginOfError = n <= 2 ? double.NaN : standardError * confidenceLevel.GetZValue(n);

            return new MeasurementsStatistics(mean, marginOfError);
        }

        static double Sum(List<double> measurements)
        {
            var sum = 0d;
            foreach (var m in measurements)
            {
                sum += m;
            }

            return sum;
        }

        static void SumWithoutOutliers(OutlierMode outlierMode, List<double> measurements,
            double lowerFence, double upperFence, out double sum, out int n)
        {
            sum = 0;
            n = 0;

            foreach (var m in measurements)
            {
                if (!IsOutlier(outlierMode, m, lowerFence, upperFence))
                {
                    sum += m;
                    ++n;
                }
            }
        }

        static double Variance(List<double> measurements, int n, double mean)
        {
            if (n == 1)
            {
                return 0;
            }

            double variance = 0;
            foreach (var m in measurements)
            {
                variance += (m - mean) * (m - mean) / (n - 1);
            }

            return variance;
        }

        static double VarianceWithoutOutliers(OutlierMode outlierMode, List<double> measurements, int n, double mean, double lowerFence, double upperFence)
        {
            if (n == 1)
            {
                return 0;
            }

            double variance = 0;
            foreach (var m in measurements)
            {
                if (!IsOutlier(outlierMode, m, lowerFence, upperFence))
                {
                    variance += (m - mean) * (m - mean) / (n - 1);
                }
            }

            return variance;
        }

        static double GetQuartile(List<double> measurements, int count)
        {
            if (count % 2 == 0)
            {
                return (measurements[count / 2 - 1] + measurements[count / 2]) / 2;
            }

            return measurements[count / 2];
        }

        static bool IsOutlier(OutlierMode outlierMode, double value, double lowerFence, double upperFence)
        {
            switch (outlierMode)
            {
                case OutlierMode.DontRemove:
                    return false;
                case OutlierMode.Remove:
                    return value < lowerFence || value > upperFence;
                default:
                    throw new ArgumentOutOfRangeException(nameof(outlierMode), outlierMode, "Unknown OutlierMode value.");
            }
        }
    }
}
