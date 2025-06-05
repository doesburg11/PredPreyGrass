# Test Attributes

This section contains a reference for attributes supported by the Performance Testing Package.

**[Performance]** - Use this with  `Test` and `UnityTest` attributes. It will initialize necessary test setup for performance tests.

**[Test]** -  Non-yielding test. This type of test starts and ends within the same frame.

**[UnityTest]** - Yielding test. This is a good choice if you want to sample measurements across multiple frames. For more on the difference between `[Test]` and `[UnityTest]`, see the Unity Test Framework [documentation](https://docs.unity3d.com/Packages/com.unity.test-framework@1.1/manual/reference-attribute-unitytest.html).

**[Version(string version)]** - Performance tests should be versioned with every change. If not specified, the default used is 1. This is essential when comparing results as results will vary any time the test changes.
