# Changelog

## [3.1.0] - 2025-03-28
### Added
- Added an optional command-line argument "perfTestResults" to control the target location of performance test run results file.
### Fixed
- Warmup cycles no longer record GC measurements.
- Setup and Cleanup cycles no longer contribute to GC measurements.

## [3.0.3] - 2023-09-11
### Fixed 
- Fixed issue where exception in OnTestEnded callback would result in EndTest method not finalising properly
### Changed
- Temporarily removed "Open Script" from Performance Benchmark Window
- Some clarifications in documentation were added ("Extension" naming changed to "Package", Package limitations clarified)

## [3.0.2] - 2023-06-29
### Changed
- Added additional InternalsVisibleTo attribute for internal test assembly

## [3.0.1] - 2023-06-23
### Changed
- Removed false "unityRelease" field from package.json

## [3.0.0] - 2023-06-05
### Added
- "Open Source Code" menu item to test results
- Test Filter to filter results by test name
### Changed
- Items in test results are grouped by classname
- Make Sample Groups to be shown only when clicking on items with them 
- "New Data available" label re-located 
- Replaced CamelCase headers with regular text headers in csv report
- Replaced TRUE/FALSE with Yes/No in the "Increase Is Better" column in csv report
- Renamed the default report name to PerformanceTestResults
- Documentation updates to reflect API changes
### Fixed
- Median calculation in case of even number of samples 
- Deviation division by zero
- Exception thrown after clicking Cancel button (during Test Result export)
- The issue where the last column was always empty in csv report

## [3.0.0-pre.2] - 2023-04-06
### Added
- Help button, which redirects the user to documentation website
- Clear Results button, which clears all Performance test results
- Updated CI to support more Unity versions and expand test coverage
### Changed
- Updated the style of toolbar buttons to match that of the Test Runner window for consistency
- Export button is now disabled instead of hidden when there are no Performance test results
- Made AutoRefresh toggle retain its state after closing and reopening the window
- Moved the Performance Test Report from 'Window/Analysis' to 'Window/General' near Test Runner for better accessibility
### Removed
- Build project from CI
### Fixed
- Issue where running tests with the Test Report window open would cause the error message "The object of type 'Material' has been destroyed" to appear
- Issues where incorrect labels were displayed at certain scenarios
- Issues flagged by SonarQube

## [3.0.0-pre.1] - 2023-03-02
### Added
- Merged 2.8.1 changes that weren't reflected in 2.10.0 release
### Fixed
- Fixing issues from SonarQube check
- Updating obsolete API's that stopped working with recent Unity versions
- Clarified how to add package in Unity project in documentation

## [2.10.0] - 2021-11-01
### Added
- Support for dynamic measurement count in Measure.Method and Measure.Frames

## [2.9.0] - 2021-04-14
### Added
- Support for overriding measurement count

## [2.8.1] - 2021-03-16
### Removed
- Setting up date when building player. It will set up at the beginning of the run.

## [2.8.0] - 2021-03-16
### Added
- Overloads to measurements for overriding sample unit
### Fixed
- Cases where cleanup throws an exception

## [2.7.0] - 2021-02-19
### Changed
- Reduce metadata overhead when running locally by caching dependencies
- Restructured documentation
### Removed
- The need for link.xml
### Fixed
- Method measurement IterationsPerMeasurement

## [2.6.0] - 2021-01-12
### Added
- Build configuration support

## [2.5.1] - 2021-01-05
### Fixed
- Serialization for Performance Test Report window

## [2.5.0] - 2020-12-29
### Added
- Domain reload support
### Changed
- Switched from Newtonsoft.Json to Unity json module

## [2.4.1] - 2020-11-05
### Changed
- Metadata collection was made public

## [2.4.0] - 2020-09-16
### Added
Performance Test Report window updates:
- CSV export option.
- Monitoring of results file timestamp to support auto refresh when a new file is found.
- Display of timestamp of last loaded results file.
- Option to sort test report window by order the tests ran in (index). This is now the default.
- Min and max to the table.
- Improved titles and tooltips on columns
### Changed
- Upgraded json dependency to release version
- Reduced overhead introduced when running tests

## [2.3.1] - 2020-07-01
### Fixed
- Overhead introduced with Measure.Method no longer calculates execution time of Setup and Cleanup changes

## [2.3.0] - 2020-06-17
### Fixed
- Measure.Method overhead
- Measure.Method no longer calculates execution time of Setup and Cleanup
- Overwritten test name will be displayed with method name in Test Result viewer

## [2.2.0] - 2020-05-26
### Added
- Support for custom metadata

## [2.1.0] - 2020-05-14
### Added
- Flexible horizontal splitter for report window
### Fixed
- Date format

## [2.0.9] - 2020-03-23
### Fixed
- Profiler measurements for method measurements
- Throw exceptions when measuring NaN

## [2.0.8] - 2020-02-20
### Fixed
- Fix profiler marker capture when changing scenes in editor tests
- Only shift samplegroups for UI

## [2.0.7] - 2020-02-14
### Fixed
- Results parsing

## [2.0.6] - 2020-01-13
### Fixed
- Development player field

## [2.0.5] - 2020-01-13
### Changed
- Disallow multiple performance attributes
- Disallow empty samplegroup name
- Assign samplegroup name to frames measurements

## [2.0.4] - 2019-12-05
### Changed
- Update json package to support AOT platforms

## [2.0.3] - 2019-11-20
### Added
- New fields to data format BuildTarget, StereoRenderingPath

## [2.0.2] - 2019-11-20
### Changed
- Increased test serialization version

## [2.0.1] - 2019-11-20
### Fixed
- Player callbacks when no tests were executed

## [2.0.0] - 2019-11-19
### Added
- Tests to package testables
### Changed
- Refactored data format, reduced nesting
- Slight refactor on measurement API
- Shift sample units when printing results
- Switched to newtosoft json package
### Removed
- Unused fields
- Deprecated attributes
### Fixed
- Resources cleanup meta files

## [1.3.1] - 2019-11-05
### Fixed
- Warning after cleaning resources
- Test suite when running in the editor

## [1.3.0] - 2019-08-26
### Changed
- Switch to errors from exceptions when parsing results
- Increase minimum unity version to 2019.3
### Removed
- Metadata collectors tests

## [1.2.6] - 2019-08-22
### Changed
- Categorize performance tests as performance
- ProfilerMarkers can now be called with string params
- Switch measuring frames and methods to stopwatch
### Removed
- Profiler section on docs as the feature was removed

## [1.2.5] - 2019-06-17
### Added
- Test publish for CI

## [1.2.4] - 2019-06-17
### Added
- Test publish for CI

## [1.2.3] - 2019-06-14
### Changed
- Updated changelog

## [1.2.2] - 2019-06-13
### Added
- Support for domain reload

## [1.2.1] - 2019-06-07
### Fixed
- Bug that would cause player build failures

## [1.2.0] - 2019-05-23
### Changed
- Increase unity version to 2019.2

## [1.1.0] - 2019-05-22
### Changed
- Update assembly definition formats to avoid testables in package manifest

## [1.0.9] - 2019-05-21
### Changed
- Update scripting runtime setting for 2019.3

## [1.0.8] - 2019-03-08
### Added
- Automation test deploy

## [1.0.7] - 2019-03-08
### Added
- Automation test deploy

## [1.0.6] - 2019-03-04
### Changed
- Updated changelog

## [1.0.5] - 2019-03-04
### Added
- Conditional support for 2019.1

## [1.0.4] - 2019-02-18
### Removed
- Unnecessary meta files

## [1.0.3] - 2019-02-18
### Changed
- package.json update

## [1.0.2] - 2019-02-18
### Changed
- package.json update

## [1.0.1] - 2019-02-18
### Changed
- Updated Documentation to reflect breaking changes

## [1.0.0] - 2019-02-15
### Changed
- Refactor attributes

## [0.1.50] - 2019-01-15
### Changed
- Results paths to persistent data

## [0.1.49] - 2018-12-04
### Changed
- Revert changes to profiler and GC

## [0.1.48] - 2018-11-22
### Changed
- Doc updates and ignore GC api in editor due to api issues

## [0.1.47] - 2018-11-14
### Removed
- Debug logs

## [0.1.46] - 2018-11-14
### Fixed
- Breaking changes introduced by testrunner API rename

## [0.1.45] - 2018-11-08
### Fixed
- Breaking changes to data submodule

## [0.1.44] - 2018-11-08
### Changed
- Disable GC and update API to work around warning

## [0.1.43] - 2018-10-30
### Fixed
- Method measurements setup and cleanup

## [0.1.42] - 2018-10-15
### Added
- Button on report window to open profiler output for test
- Save profiler output on perf tests
### Removed
- Unsupported features for legacy scripting runtime
- Unnecessary assembly definition
### Fixed
- Version attribute for test cases

## [0.1.41] - 2018-10-02
### Added
- Test report graph

## [0.1.40] - 2018-09-17
### Changed
- Update documentation

## [0.1.39] - 2018-09-14
### Removed
- Duplicate module from docs

## [0.1.38] - 2018-09-14
### Changed
- Updated documentation

## [0.1.36] - 2018-08-27
### Changed
- ProfilerMarkers now take params as arguments

## [0.1.35] - 2018-08-27
### Added
Measure.Method improvements:
- Add GC allocation to Measure.Method
- Add setup/cleanup for Measure.Method
- Move order of calls for Measure.Scope

## [0.1.34] - 2018-08-16
### Fixed
- Obsolete warnings

## [0.1.33] - 2018-08-03
### Fixed
- Obsolete warnings, doc update with modules and internals, ValueSource fix

## [0.1.32] - 2018-07-09
### Added
- Method and Frames measurements can now specify custom warmup, measurement and iteration counts

## [0.1.31] - 2018-07-04
### Changed
- Marked metadata tests with performance category

## [0.1.30] - 2018-06-27
### Fixed
- Method measurement

## [0.1.29] - 2018-06-12
### Changed
- Moving back to json in xml due to multiple instabilities

## [0.1.28] - 2018-06-01
### Removed
- json printing from output

## [0.1.27] - 2018-05-31
### Added
- Meta files to npm ignore

## [0.1.26] - 2018-05-31
### Changed
Preparing package for moving to public registry:
- Inversed changelog order
- Excluded CI files from published package

## [0.1.25] - 2018-05-31
### Removed
- Missing meta files

## [0.1.24] - 2018-05-31
### Changed
- Print out json to xml by default for backwards compatability

## [0.1.23] - 2018-05-30
### Fixed
- Issues with packman, bumping up version

## [0.1.22] - 2018-05-29
### Added
- Option to specify custom Measure.Method Execution and Warmup count

## [0.1.21] - 2018-05-25
### Fixed
- Issues introduced by .18 fix

## [0.1.19] - 2018-05-24
### Changed
- Package has been renamed to `com.unity.test-framework.performance` to match test framework

## [0.1.18] - 2018-05-24
### Fixed
- Fix SetUp and TearDown for 2018.1

## [0.1.17] - 2018-05-23
### Changed
- Refactor Method and Frames measurements
- Metadata collected using internal test runner API and player connection for 2018.3+

## [0.1.16] - 2018-05-09
### Fixed
- Bug fix regarding measureme methods being disposed twice

## [0.1.15] - 2018-05-02
### Fixed
- Metadata test, the test was failing if a json file was missing for playmode tests

## [0.1.14] - 2018-04-30
### Added
- Addition of measuring a method or frames for certain amount of times or for duration
- Introduced SampleGroupDefinition
### Changed
- Refactored measuring methods
### Removed
- Removes linq usage for due to issues with AOT platforms

## [0.1.13] - 2018-04-15
### Added
- Added total, std and sample count aggregations
- Added sample unit to multi sample groups
### Removed
- Removed totaltime from frametime measurements
### Fixed
- Fixed android metadata collecting

## [0.1.12] - 2018-04-11
### Changed
- Naming
### Fixed
- json serialization

## [0.1.11] - 2018-04-09
### Fixed
- 2018.1 internal namespaces

## [0.1.10] - 2018-04-09
### Added
- Added editmode and playmode tests that collect metadata
### Changed
- Change fields to UpperCamelCase

## [0.1.9] - 2018-04-06
### Added
- json output for 2018.1 which will be printed after test run

## [0.1.8] - 2018-04-03
### Fixed
- Fix an exception on 2018.1

## [0.1.7] - 2018-04-03
### Changed
- Changed some of the names to match new convention
- Addressed typos in docs
- Multiple overloads replaced by using default arguments

## [0.1.6] - 2018-03-28
### Added
- Measure.Custom got a new overload with SampleGroup
- Readme now includes installation and more examples

## [0.1.5] - 2018-03-20
### Added
- Checks for usage outside of Performance tests

## [0.1.4] - 2018-03-20
### Added
- System info to performance test output
- Preparing for reporting test data

## [0.1.3] - 2018-03-14
### Removed
- Temporarily removing tests from the package into separate repo

## [0.1.2] - 2018-03-14
### Fixed
- Update for a missing bracket

## [0.1.1] - 2018-03-14
### Added
- Test output now includes json that can be used to parse performance data from TestResults.xml
- Added defines to be compatible with 2018.1 and newer
- Measurement methods can now take in SampleGroup as argument
### Removed
- Removed unnecessary overloads for measurements due to introduction of SampleGroup

## [0.1.0] - 2018-02-27
### This is the first release of *Unity Package performancetesting*.
Initial version.
