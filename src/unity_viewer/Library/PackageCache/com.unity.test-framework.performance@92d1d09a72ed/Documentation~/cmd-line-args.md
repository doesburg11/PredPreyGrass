# Command-line arguments

This section lists Unity arguments supported by the package when running performance tests from the command line. For a general overview of the topic, see the [Unity Test Framework documentation](https://docs.unity3d.com/Packages/com.unity.test-framework@latest).

## Supported arguments

### perfTestResults

The path where Unity should save the JSON file with performance test run results. By default, Unity saves it in the [Application.persistentDataPath](https://docs.unity3d.com/ScriptReference/Application-persistentDataPath.html) folder.

Please note that when this argument is not provided, Unity generates two report files at the default results location - the JSON file with performance test run results as well as the NUnit test results XML file that these results are extracted from. This behaviour currently exists for backwards compatibility reasons and will be deprecated over time, as the destination path of the test results XML file is already controlled by the `testResults` command-line argument of the Unity Test Framework package.

## Example of command-line usage

```bash
Unity.exe -runTests -batchmode -projectPath PATH_TO_YOUR_PROJECT -testResults C:\temp\results.xml -perfTestResults C:\temp\perfResults.json
```
