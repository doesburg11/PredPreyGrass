# About Unity Test Framework

The Unity Test Framework (UTF) enables Unity users to test their code in both **Edit Mode** and **Play Mode**, and also on target platforms such as [Standalone](https://docs.unity3d.com/Manual/Standalone.html), Android, iOS, etc.

This package provides a standard test framework for users of Unity and developers at Unity so that both benefit from the same features and can write tests the same way. 

UTF uses a Unity integration of NUnit library, which is an open-source unit testing library for .Net languages. UTF currently uses NUnit version 3.5. For more information about NUnit, see the [official NUnit website](http://www.nunit.org/) and the [NUnit documentation](https://docs.nunit.org/).

> **Note**: UTF is not a new concept or toolset; it is an adjusted and more descriptive naming for the toolset otherwise known as Unity Test Runner, which is now available as this package. 

# Installing Unity Test Framework

The Test Framework package is shipped with the Unity Editor and should be automatically included in any project created with Unity 2019.2 or later. If you need to install the package manually, you can do so in any of the standard ways documented in the [Package Manager documentation](https://docs.unity3d.com/Packages/com.unity.package-manager-ui@latest/index.html).

> **Note**: If you're adding the package by name, the canonical name to use is `com.unity.test-framework`. If you're adding it from the registry, the package is listed under the display name `Test Framework`. 

# Using Unity Test Framework

To learn how to use the Unity Test Framework package in your project, read the [manual](./manual.md).

# Technical details

## Requirements

This version of the Unity Test Framework is compatible with the following versions of the Unity Editor:

* 2019.2 and later.

## Known limitations

Unity Test Framework version 1.3.x includes the following known limitations:

* The `UnityTest` attribute does not support WSA platform.
* The `UnityTest` attribute does not support [Parameterized tests](https://github.com/nunit/docs/wiki/Parameterized-Tests) (except for `ValueSource`).
* The `UnityTest` attribute does not support the `NUnit` [Repeat](https://github.com/nunit/docs/wiki/Repeat-Attribute) attribute.
* Nested test fixture cannot run from the Editor UI. 
* When using the `NUnit` [Retry](https://github.com/nunit/docs/wiki/Retry-Attribute) attribute in PlayMode tests, it throws `InvalidCastException`.