# Aseprite Features
This page highlights which Aseprite feature the Aseprite Importer supports/does not support.

## Supported features
**File formats**
- .ase & .aseprite
- Color modes (All modes are supported)
    - RGBA
    - Grayscale
    - Indexed

**Layer settings**
- Visible/Hidden layer
    - Hidden layers are not imported by default. This can be changed by checking “Include hidden layers” in the import settings.
- Layer blend modes
    - All blend modes are supported with Import Mode: Merge frames.
- Layer & Cell opacity
- Linked Cells
- Tags
    - Only Animation Direction: Forward is supported.
    - Values set in the repeating field only have two results on import:
        - ∞ will result in a looping Animation Clip. This value is the default for all Tags in Aseprite.
        - 1 -> N will result in a non looping Animation Clip.
- [Individual frame timings](https://www.aseprite.org/docs/frame-duration/)
- [Layer groups](https://www.aseprite.org/docs/layer-group/)
    - The importer respects the visibility mode selected for the group. If a group is hidden, underlying layers will not be imported by default.
    - Layer groups will be generated in the prefab hierarchy if the import mode is set to **Individual layers**.
- User data
    - The user data field in Cells can be used to inject events to a generated Animation Clip. Read more [here](./ImporterFAQ.md#how-to-add-events-to-animation-clips).

**Misc**
- [Tilemaps](https://www.aseprite.org/docs/tilemap/)
    - When importing tile data, select **Import Mode: Tile Set**. This will tell Unity to generate [Tile assets](https://docs.unity3d.com/Manual/Tilemap-TileAsset.html) for each Aseprite Tile, and a [Tile Palette](https://docs.unity3d.com/Manual/Tilemap-Palette.html) to store them all in.
    - The Aseprite Importer only supports Tile Sets which are of the same size. If an Aseprite file is imported with multiple different Tile Set sizes, the first used Tile Set will be used to setup the Tile sizes inside of Unity.

## Unsupported features
- [Slices](https://www.aseprite.org/docs/slices/)
