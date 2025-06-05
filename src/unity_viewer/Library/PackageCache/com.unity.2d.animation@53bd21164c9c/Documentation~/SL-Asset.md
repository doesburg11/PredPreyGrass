# Sprite Library Asset in Unity

The **Sprite Library Asset** is an Unity asset that contains the Sprites that you want to use for [Sprite Swapping](SpriteSwapIntro.md). This page explains what are the [Sprite Library Asset properties](#sprite-library-asset-properties) and how to [create a Sprite Library Asset](#create-a-sprite-library-asset) or a [Sprite Library Asset variant](#convert-a-sprite-library-asset-into-a-variant).

A Sprite Library Asset groups the Sprites it contains into [Categories](SL-Editor.md#categories), and you can give these Sprites unique names called [Labels](SL-Editor.md#labels) to differentiate them. You can edit the Sprite Library Asset's content in the [Sprite Library Editor window](SL-Editor.md) (refer to its documentation for more details).

In the [Sprite Swap workflow](SpriteSwapSetup.md), after creating a Sprite Library Asset or several assets, you can select the Sprite Library Asset you want to use with the [Sprite Library](SL-component.md) component, and the [Sprite Resolver](SL-Resolver.md) component will pull information from the asset selected.

## Create a Sprite Library Asset

To create a Sprite Library Asset, go to **Assets** > **Create** > **2D** > **Sprite Library Asset**.

![](images/2D-animation-SLAsset-dropdown.png)

## Sprite Library Asset properties
Select the Sprite Library Asset and go to its Inspector window to view the following properties.

![](images/2D-animation-SLAsset-properties.png)

Property  |Description
--|--
**Open in Sprite Library Editor**  |   Select this to open the [Sprite Library Editor window](SL-Editor.md) to edit the content of this asset.
**Main Library**  |  Leave this property empty to have this Sprite Library Asset refer to its own Categories and Labels. Assign a different Sprite Library Asset to have it become the [Main Library](SL-Editor-UI.md#main-library) of the selected Sprite Library Asset, which will now refer to the second asset's Categories and Labels instead. Doing so also [converts](#convert-a-sprite-library-asset-into-a-variant) the selected Sprite Library Asset into a Variant asset of the Sprite Library Asset set as the **Main Library**.
**Revert**  |  Select this to reset property changes back to the last saved state. Selecting this removes all unsaved changes.
**Apply**  |  Select this to save the current property settings.

## Create a Sprite Library Asset Variant

A Sprite Library Asset Variant inherits **Categories** and **Labels** from a selected Sprite Library Asset, instead of referring to its own. There are two ways to create a Variant.

### Create through the menu

After creating a Sprite Library Asset, select it in the Project window, then go to **Assets** > **Create** > **2D** > **Sprite Library Asset Variant** to create a Variant asset that references it.

![](images/2D-animation-SLAssetVariant-dropdown.png)

### Convert a Sprite Library Asset into a Variant

You can convert an existing Sprite Library Asset into a Variant of another by setting another Sprite Library Asset as its [Main Library](SL-Main-Library.md).

## Additional resources
- [Sprite Library Editor](SL-Editor.md)
- [Setting up for Sprite Swap](SpriteSwapSetup.md)
- [Overrides to the Main Library](SL-Main-Library.md)
