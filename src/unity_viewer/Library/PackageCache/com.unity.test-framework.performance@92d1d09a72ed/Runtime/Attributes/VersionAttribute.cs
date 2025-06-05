using System;
using NUnit.Framework;
using NUnit.Framework.Interfaces;
using NUnit.Framework.Internal;

namespace Unity.PerformanceTesting
{
    /// <summary>
    /// Test attribute to specify test version.
    /// </summary>
    [AttributeUsage(AttributeTargets.Method | AttributeTargets.Class)]
    public class VersionAttribute : NUnitAttribute, IApplyToTest
    {
        /// <summary>
        /// Test version.
        /// </summary>
        public string Version { get; }

        /// <summary>
        /// Adds attribute to specify test version.
        /// </summary>
        /// <param name="version">Version of the test.</param>
        public VersionAttribute(string version)
        {
            Version = version;
        }

        /// <summary>
        /// Used by NUnit to apply version to properties.
        /// </summary>
        /// <param name="test">An NUnit test to apply the version property to.</param>
        public void ApplyToTest(Test test)
        {
            test.Properties.Add("Version", this);
        }
    }
}
