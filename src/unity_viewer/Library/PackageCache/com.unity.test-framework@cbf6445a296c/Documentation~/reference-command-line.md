# Running tests from the command line

It’s pretty simple to run a test project from the command line. Here is an example in Windows:

```bash
Unity.exe -runTests -batchmode -projectPath PATH_TO_YOUR_PROJECT -testResults C:\temp\results.xml -testPlatform PS4
```

> **Note**: Use the `-batchmode` option when running tests on the command line to remove the need for manual user inputs. For more information, see Unity [Command line arguments](https://docs.unity3d.com/Manual/CommandLineArguments.html).

## Test Framework command line arguments

### forgetProjectPath

Don't save your current **Project** into the Unity launcher/hub history.

### runTests

Runs tests in the Project.

### testCategory

A semicolon-separated list of test categories to include in the run, or a regular expression pattern to match category names. A semi-colon separated list should be formatted as a string enclosed in quotation marks, e.g. `testCategory "firstCategory;secondCategory"`. If using both `testFilter` and `testCategory`, then only tests that match both are run. This argument supports negation using '!'. If using '!MyCategory' then no tests with the 'MyCategory' category will be included in the run.

### testFilter

A semicolon-separated list of test names to run, or a regular expression pattern to match tests by their full name. A semi-colon separated list should be formatted as a string enclosed in quotation marks, e.g. `testFilter "Low;Medium"`. This argument supports negation using '!'. If using the test filter '!MyNamespace.Something.MyTest', then all tests except that test will be run.
It is also possible to run a specific variation of a parameterized test like so: `"ClassName\.MethodName\(Param1,Param2\)"`

### testPlatform

The platform to run tests on. Accepted values: 

* **EditMode**
    * Edit Mode tests. Equivalent to running tests from the EditMode tab of the Test Runner window.
* **PlayMode**
    * Play Mode tests that run in the Editor. Equivalent to running tests from the PlayMode tab of the Test Runner window.
* Any value from the [BuildTarget](https://docs.unity3d.com/ScriptReference/BuildTarget.html) enum.
    * Play Mode tests that run on a player built for the specified platform. Equivalent to using the **Run all tests (`<target_platform>`)** dropdown in the PlayMode tab of the Test Runner window.

> **Note**: If no value is specified for this argument, tests run in Edit Mode.

### assemblyNames

A semicolon-separated list of test assemblies to include in the run. A semi-colon separated list should be formatted as a string enclosed in quotation marks, e.g. `assemblyNames "firstAssembly;secondAssembly"`.

### testResults

The path where Unity should save the result file. By default, Unity saves it in the Project’s root folder. Test results follow the XML format as defined by NUnit, see the [NUnit documentation](https://docs.nunit.org/articles/nunit/technical-notes/usage/Test-Result-XML-Format.html). There is currently no common definition for exit codes reported by individual Unity components under test. The best way to understand the source of a problem is the content of error messages and stack traces.

### playerHeartbeatTimeout

The time, in seconds, the editor should wait for heartbeats after starting a test run on a player. This defaults to 10 minutes.

### runSynchronously

If included, the test run will run tests synchronously, guaranteeing that all tests runs in one editor update call. Note that this is only supported for EditMode tests, and that tests which take multiple frames (i.e. `[UnityTest]` tests, or tests with `[UnitySetUp]` or `[UnityTearDown]` scaffolding) will be filtered out.

### testSettingsFile 

Path to a [TestSettings.json](./reference-test-settings-file.md) file.

### orderedTestListFile

Path to a *.txt* file (which can have any name as long as the format is text) which contains a list of full test names you want to run in the specified order. The tests should be seperated by new lines and if they have parameters, these should be specified as well. The following is an example of the content of such a file:

```
Unity.Framework.Tests.OrderedTests.NoParameters
Unity.Framework.Tests.OrderedTests.ParametrizedTestA(3,2)
Unity.Framework.Tests.OrderedTests.ParametrizedTestB("Assets/file.fbx")
Unity.Framework.Tests.OrderedTests.ParametrizedTestC(System.String[],"foo.fbx")
Unity.Framework.Tests.OrderedTests.ParametrizedTestD(1.0f)
Unity.Framework.Tests.OrderedTests.ParametrizedTestE(null)
Unity.Framework.Tests.OrderedTests.ParametrizedTestF(False, 1)
Unity.Framework.Tests.OrderedTests.ParametrizedTestG(float.NaN)
Unity.Framework.Tests.OrderedTests.ParametrizedTestH(SomeEnum)
```

### randomOrderSeed
An integer different from 0 that set the seed to randomize the tests in the project, indipendetly from the fixture. 
```
# normal order 
Test_1 
Test_2 
Test_3 
Test_4
# randomized with seed x 
Test_3 
Test_1 
Test_4 
Test_2
# randomized with same seed x and a new test 
Test_3 
Test_5
Test_1 
Test_4 
Test_2
```


### retry

An integer that sets the retry count. Failing tests will be retried up to this number of times, or until they succeed, whichever happens first.

### repeat

An integer that set the repeat count. Successful tests will be repeated up to this number of times, or until they fail, whichever happens first.