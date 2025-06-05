# 2D Sprite package

Install the 2D Sprite package to install the Sprite Editor, which allows you to create and edit Sprite assets. The Sprite Editor Data Provider API also allow user extensibility to add custom behaviour for editing various Sprite related data. To install the package, search for it in the Package Manager window and [install it from the registry](xref:upm-ui-install). If you created your Project with the [2D Template](xref:Quickstart2DSetup), this package is automatically installed.

This version of Sprite Editor is compatible with the following versions of the Unity Editor:

* 2019.4 and later (recommended)

| **Topic**             | **Description**         |
| :-------------------- | :----------------------- |
| [Sprite Editor (User Manual)](xref:SpriteEditor)  | Understand how to use the main features of the Sprite Editor.   |
| [Sprite Editor Data Provider APIs](DataProvider.md)    | Understand how to use the APIs available edit Sprite data.    |

## Package contents

The following table indicates the folder structure of the Sprite package:

| **Location** | **Description** |
|---|---|
|`<Editor>`|Root folder containing the source for the Sprite Editor.|
|`<Tests>`|Root folder containing the source for the tests for Sprite Editpr used the Unity Editor Test Runner.|

## Documentation revision history

| **Date** | **Reason** |
|---|---|
|April 13, 2022|Added Sprite Editor Data Provider API samples|
|January 25, 2019|Document created. Matches package version 1.0.0|

## Additional resources

* [Sprite Editor: Custom Outline](xref:SpriteOutlineEditor)
* [Sprite Editor: Custom Physics Shape](xref:CustomPhysicsShape)
* [Sprite Editor: Secondary Textures](xref:SpriteEditor-SecondaryTextures)
