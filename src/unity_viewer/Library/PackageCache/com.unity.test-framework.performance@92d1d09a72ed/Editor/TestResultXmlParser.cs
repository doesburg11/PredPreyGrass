using System;
using Unity.PerformanceTesting.Data;
using UnityEngine;

namespace Unity.PerformanceTesting.Editor
{
    /// <summary>
    /// Helper class to parse test runs into performance test runs.
    /// </summary>
    public class TestResultXmlParser
    {
        /// <summary>
        /// Parses performance test run from test run result xml.
        /// </summary>
        /// <param name="resultXmlFileName">Path to test results xml file.</param>
        /// <returns>Performance test run data extracted from the NUnit xml results file.</returns>
        public Run GetPerformanceTestRunFromXml(string resultXmlFileName)
        {
            return TestResultsParser.GetPerformanceTestRunDataFromXmlFile(resultXmlFileName);
        }
    }
}
