# 7\. PlayMode tests

## Learning objectives

This exercise introduces the concept of Play Mode tests and will teach you:

*   When to use Play Mode tests.
*   How to set up an assembly definition for PlayMode tests.
*   How to run tests in Play Mode.

## Intro and motivation

Managed code in Unity generally exists in two different modes: Edit Mode and Play Mode. Edit Mode is code executing inside the Editor, which includes things like our UIs and underlying logic. Play Mode is when the game or 3D application is playing, which is either when the user presses the play button in the Editor or when the code runs in a standalone player.  
  
We have distinct tests for each mode because how they run and what they can access is different in each case. Methods from a given API may only be available in one mode. Due to this distinction, tests for Edit Mode and Play Mode are in different assemblies.  
  
You can create a Play Mode test assembly by following the instructions in the Play Mode tab of the Test Runner UI. Detailed instructions are available in the [Getting started section](../workflow-create-test.md). The difference in the assembly definition between Edit Mode and Play Mode is what platforms they are enabled for. An Edit Mode test assembly is only enabled for the `Editor` platform. Enabling any other platforms automatically makes it a Play Mode test assembly, as tests can now run on other platforms. By default, Play Mode tests are set to run on all platforms.

## Exercise

The [sample](./welcome.md#import-samples) `7_PlayModeTests` contains an empty project. Import this sample and add a new assembly for Play Mode tests.

Afterwards add a test which just asserts that `Application.isPlaying` is true. This flag will only be true when in Play Mode.  

Run the test. Notice that your Editor enters Play Mode (the equivalent of pressing the play button) while the test is running and exits Play Mode afterward.

## Solution

A full solution with the test and assembly setup is available in the `7_PlayModeTests_Solution` sample.

## Further reading and resources

[EditMode vs PlayMode](https://docs.unity3d.com/Packages/com.unity.test-framework@1.1/manual/edit-mode-vs-play-mode-tests.html)
