using System;

namespace Unity.PerformanceTesting.Exceptions
{
    /// <summary>
    /// Performance test exception.
    /// </summary>
    [Serializable]
    public class PerformanceTestException : System.Exception
    {
        /// <summary>
        /// Performance test exception. Used to indicate failures while running a performance test.
        /// </summary>
        /// <param name="message">Exception message.</param>
        public PerformanceTestException(string message)
            : base(message) { }
    }
}
