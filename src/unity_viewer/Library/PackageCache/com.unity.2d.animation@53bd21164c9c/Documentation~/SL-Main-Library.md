# Overrides to the Main Library

When you create a [Sprite Library Asset](SL-Asset.md), you have the option of [converting it into a Sprite Library Asset Variant](SL-Asset.md#convert-a-sprite-library-asset-into-a-variant), or [creating a Variant asset](SL-Asset.md#create-a-sprite-library-asset-variant) of the selected Sprite Library Asset. A Variant asset inherits all the Categories and Labels from the Sprite Library Asset as its **Main Library**. You can't directly change the Categories and Labels inherited from the Main Library, but you can add to the inherited content by adding new Categories and Labels in the form of [overrides](#create-overrides).

## Inherited Categories and Labels limitations

Assigning another existing Sprite Library Asset to the **Main Library** property of the current Sprite Library Asset allows the current Asset to access all Categories and Labels contained in the assigned Sprite Library Asset. Inherited Categories are visible in the [Sprite Library Editor window](SL-Editor-UI.md) in the [Inherited foldout group](SL-Editor-UI.md#categories-and-labels-columns), while the **Local** foldout group contains all Categories that exist solely in the current asset.

You can't rename or remove the Labels of inherited Categories, but you can add new Labels to an inherited Category as an [override](#create-overrides).

## Create overrides

An override is any change you make to the contents of an inherited Category. While you can't rename or remove inherited Labels, you can do the following in an inherited Category:

- [Create a new Label](SL-Editor.md#create-a-label).
- [Change the sprite](#change-the-sprite-reference) that the Label references.
- Revert overrides in selected Labels or for all inherited Categories and Labels.

The Sprite Library Editor window [displays a vertical white line](SL-Editor-UI.md#variant-asset-specific-properties) next to inherited Categories and inherited Labels when an override is present.

### Change the sprite reference

You can change a Label's sprite reference by selecting a different sprite from the object picker next to the [Sprite object field](SL-Editor-UI.md#categories-and-labels-columns). You can also change the sprite reference by [dragging](SL-Drag.md) the desired sprite directly onto to a Label.

To revert sprite reference changes made to selected Labels, right-click the Label(s) and select **Revert Selected Overrides** from the [context menu](SL-Editor-UI.md#label-context-menu) to restore all sprite references back to their original inherited state from the Main Library.

![](images/2D-animation-SLAsset-label-revert.png)<br/>_Revert changes to Labels in inherited Categories by selecting **Revert Selected Overrides**._

To revert all overrides in the selected inherited Category, select **Revert All Overrides** from the context menu.

**Caution:** Overrides aren't included in the [save state](SL-Editor-UI.md#saving-options) of the Sprite Library Editor, and reverting overrides will remove all overrides regardless of the previous save state. To undo the last action, press Ctrl+Z (macOS: Cmd+Z).