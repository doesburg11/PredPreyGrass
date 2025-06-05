using System;

namespace Unity.PerformanceTesting
{
    /// <summary>
    /// Measurement unit used for sample groups.
    /// </summary>
    public enum SampleUnit
    {
        /// <summary>
        /// Nanoseconds.
        /// </summary>
        Nanosecond,

        /// <summary>
        /// Microseconds.
        /// </summary>
        Microsecond,

        /// <summary>
        /// Milliseconds.
        /// </summary>
        Millisecond,

        /// <summary>
        /// Seconds.
        /// </summary>
        Second,

        /// <summary>
        /// Bytes.
        /// </summary>
        Byte,

        /// <summary>
        /// Kilobytes.
        /// </summary>
        Kilobyte,

        /// <summary>
        /// Megabytes.
        /// </summary>
        Megabyte,

        /// <summary>
        /// Gigabytes.
        /// </summary>
        Gigabyte,

        /// <summary>
        /// Undefined, represents any other unit we don't have by default. When using it make sure your sample group name represents the measurement.
        /// </summary>
        Undefined
    }

    static class SampleUnitExtensions
    {
        public static string ShortName(this SampleUnit s)
        {
            switch (s)
            {
                case SampleUnit.Nanosecond:
                    return "ns";
                case SampleUnit.Microsecond:
                    return "μs";
                case SampleUnit.Millisecond:
                    return "ms";
                case SampleUnit.Second:
                    return "s";
                case SampleUnit.Byte:
                    return "b";
                case SampleUnit.Kilobyte:
                    return "kb";
                case SampleUnit.Megabyte:
                    return "mb";
                case SampleUnit.Gigabyte:
                    return "gb";
                case SampleUnit.Undefined:
                    return "";
                default:
                    throw new ArgumentOutOfRangeException(nameof(s), s, null);
            }
        }
    }
}
