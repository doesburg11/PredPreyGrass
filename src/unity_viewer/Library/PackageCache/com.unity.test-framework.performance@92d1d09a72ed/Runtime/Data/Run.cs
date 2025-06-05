using System;
using System.Collections.Generic;

namespace Unity.PerformanceTesting.Data
{
    /// <summary>
    /// Represents a performance test run.
    /// </summary>
    [Serializable]
    public class Run
    {
        /// <summary>
        /// Test Suite.
        /// </summary>
        [RequiredMember] public string TestSuite;
        /// <summary>
        /// Test run start datetime in Unix Epoch milliseconds format.
        /// </summary>
        [RequiredMember] public long Date;
        /// <summary>
        /// Player settings.
        /// </summary>
        [RequiredMember] public Player Player;
        /// <summary>
        /// Hardware information.
        /// </summary>
        [RequiredMember] public Hardware Hardware;
        /// <summary>
        /// Editor version.
        /// </summary>
        [RequiredMember] public Editor Editor;
        /// <summary>
        /// Package dependencies.
        /// </summary>
        [RequiredMember] public List<string> Dependencies = new List<string>();
        /// <summary>
        /// List of performance test results.
        /// </summary>
        [RequiredMember] public List<PerformanceTestResult> Results = new List<PerformanceTestResult>();
    }
}
