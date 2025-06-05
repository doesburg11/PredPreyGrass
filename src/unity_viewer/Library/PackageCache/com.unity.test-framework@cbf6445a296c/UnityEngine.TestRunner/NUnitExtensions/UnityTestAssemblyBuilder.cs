using System;
using System.Collections.Generic;
using System.IO;
using System.Reflection;
using NUnit;
using NUnit.Framework.Api;
using NUnit.Framework.Interfaces;
using NUnit.Framework.Internal;
using Unity.Profiling;
using UnityEngine.TestRunner.NUnitExtensions;
using UnityEngine.TestRunner.NUnitExtensions.Filters;

namespace UnityEngine.TestTools.NUnitExtensions
{
    internal class UnityTestAssemblyBuilder : DefaultTestAssemblyBuilder, IAsyncTestAssemblyBuilder
    {
        private readonly string m_ProductName;
        private readonly ITestSuiteModifier[] m_TestSuiteModifiers;

        public UnityTestAssemblyBuilder(string[] orderedTestNames, int randomSeed)
        {
            m_TestSuiteModifiers = (orderedTestNames != null && orderedTestNames.Length > 0) || randomSeed != 0
                ? new ITestSuiteModifier[] {new OrderedTestSuiteModifier(orderedTestNames, randomSeed)}
                : new ITestSuiteModifier[0];
            m_ProductName = Application.productName;
        }

        public ITest Build(Assembly[] assemblies, TestPlatform[] testPlatforms, IDictionary<string, object> options)
        {
            var test = BuildAsync(assemblies, testPlatforms, options);
            while (test.MoveNext())
            {
            }

            return test.Current;
        }
        
        struct PlatformAssembly : IEquatable<PlatformAssembly>
        {
            public System.Reflection.Assembly Assembly;
            public TestPlatform Platform;

            public bool Equals(PlatformAssembly other)
            {
                return Equals(Assembly, other.Assembly) && Platform == other.Platform;
            }

            public override bool Equals(object obj)
            {
                return obj is PlatformAssembly other && Equals(other);
            }

            public override int GetHashCode()
            {
                unchecked
                {
                    return ((Assembly != null ? Assembly.GetHashCode() : 0) * 397) ^ (int) Platform;
                }
            }
        }

        private static Dictionary<PlatformAssembly, TestSuite> CachedAssemblies = new Dictionary<PlatformAssembly, TestSuite>();

        public IEnumerator<ITest> BuildAsync(Assembly[] assemblies, TestPlatform[] testPlatforms, IDictionary<string, object> options)
        {
            var productName = string.Join("_", m_ProductName.Split(Path.GetInvalidFileNameChars()));
            var suite = new TestSuite(productName);
            var lastYieldTime = Time.realtimeSinceStartup;
            for (var index = 0; index < assemblies.Length; index++)
            {
                var assembly = assemblies[index];
                var platform = testPlatforms[index];

                using (new ProfilerMarker(nameof(UnityTestAssemblyBuilder) + "." + assembly.GetName().Name).Auto())
                {
                    var key = new PlatformAssembly {Assembly = assembly, Platform = platform};
                    if (!CachedAssemblies.TryGetValue(key, out var assemblySuite))
                    {
                        assemblySuite = Build(assembly, GetNUnitTestBuilderSettings(platform)) as TestSuite;
                        if (assemblySuite != null)
                        {
                            assemblySuite.Properties.Set("platform", platform);
                            EditorOnlyFilter.ApplyPropertyToTest(assemblySuite, platform == TestPlatform.EditMode);
                        }
                        CachedAssemblies.Add(key, assemblySuite);
                    }

                    if (assemblySuite != null && assemblySuite.HasChildren)
                    {
                        suite.Add(assemblySuite);
                    }
                }

                if (Time.realtimeSinceStartup > lastYieldTime + 0.1f)
                {
                    yield return null;
                    lastYieldTime = Time.realtimeSinceStartup;
                }
            }

            suite.ParseForNameDuplicates();
            suite.Properties.Set("platform", testPlatforms.MergeFlags());

            foreach (var testSuiteModifier in m_TestSuiteModifiers)
            {
                suite = testSuiteModifier.ModifySuite(suite);
            }

            yield return suite;
        }

        public static Dictionary<string, object> GetNUnitTestBuilderSettings(TestPlatform testPlatform)
        {
            var emptySettings = new Dictionary<string, object>();
            emptySettings.Add(FrameworkPackageSettings.TestParameters, "platform=" + testPlatform);
            return emptySettings;
        }
    }
}
