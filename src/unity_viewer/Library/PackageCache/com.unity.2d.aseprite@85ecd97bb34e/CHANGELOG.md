# Changelog
## [1.2.4] - 2025-03-18
### Fixed
- Fixed an issue where going from Import Mode: Individual Layers to Import Mode: Merge Frame would cause the importer to fail the import. (DANB-875)
- Fixed an issue where the content in the Aseprite's Preference menu were misaligned. (DANB-880)
- Fixed an issue where some elements in the Aseprite Importer inspector could not be edited when in Debug mode. (DANB-863)

## [1.2.3] - 2025-03-01
### Added
- Added support for Aseprite's new Layer UUID, to help identify each Aseprite Layer by id instead of name and path.

### Fixed
- Fixed an issue where changes to a Tile's size in Aseprite would not be reflected in Unity. (DANB-816)
- Fixed an issue where Tile Layers from Aseprite would not stack in the generated Tile Palette.
- Slight improvement to the way tile data is imported, resulting in slightly quicker imports.

## [1.2.2] - 2025-02-27
### Fixed
- Fixed an issue where the Tile Palette would not be generated the same way as in Aseprite when the Aseprite file contains a Tile Layer. (DANB-824)

## [1.2.1] - 2025-01-07
### Fixed
- Fixed an issue where Sprites smaller than the tile size would be wrongly placed in the tile palette.
- Fixed an issue where the importer inspector would throw an exception if an animation clip was empty of data. (DANB-788)
- Fixed an issue where the Z-Index would not be taken into account when the import mode is set to "Merge Frame". (DANB-787)

## [1.2.0] - 2024-08-29
### Added
- Added a new import mode, Tile Set, which imports tile data from Aseprite and generates Unity Tilemap assets on import.
- Added support for a parameter in the generated animation events.
- Added a choice on how Animation Events are stored, as individual events or grouped up and listened through `OnAnimationEvent(string)`.

### Fixed
- Reduced the font size slightly in the importer headers to match other inspector headers in Unity. (DANB-644)
- Fixed an issue where the Sort Order would not be reset in Animation Clips when making use of the Z-index in Aseprite.
- Fixed an issue where SpriteRenderers would lose their reference if an Aseprite file's name was changed. (DANB-692)
- Fixed an issue where adding Sprite Bones would cause the Sprites to become corrupt in the importation. (DANB-779)
- Fixed an issue where the sprite rect would not get updated after being set in the Sprite Editor.
- Fixed an issue where the references to sprites would be lost when adding new layers to an Aseprite file. (DANB-782)

## [1.1.4] - 2024-05-02
### Fixed
- Fixed an issue where Sprite Renderers would be hidden after transitioning from one Animation Clip to another.
- Fixed an issue where generated AnimationClips would be 0.01 seconds too long.

## [1.1.3] - 2024-03-25
### Fixed
- Fixed an issue where the importer would not parse palette data from the "old palette" chunks.
- Fixed an issue where the Physics Shapes would not take the Sprite Rects into account, causing the outline to be wrongly offset.
- Fixed an issue where .ase/.aseprite files containing z-index data would fail to import. (DANB-608)

## [1.1.2] - 2024-03-10
### Fixed
- Fixed an issue where the Mosaic padding did not show up in Sprite Sheet import mode.
- Fixed an issue where using Sprite Padding with individual import mode would misalign the GameObjects in the generated model prefab.
- Fixed an issue where the Aseprite package would contest with the XR subsystem package over the InternalAPIEditorBridge.005. (UUM-49338)

## [1.1.1] - 2024-01-03
### Fixed
- Fixed an issue where the Sprite Editor could be opened even though there was no valid texture to open it with.
- Fixed an issue where the importer would not generate a square power-of-two texture for compressions which needs it (pvrtc).
- Fixed an issue where changes to linked cells would not be taken into account when reimporting.

## [1.1.0] - 2023-11-24
### Added
- Added a mosaic padding option to the importer editor.
- Added "Generate Physics Shape" option to the importer editor.

### Changed
- Fixed an issue where the background importer would act on files that were not Aseprite files.

## [1.0.0] - 2023-05-17
### Added
- Added a new event to the Aseprite Importer which is fired at the last import process step.
- Made the Aseprite file property publicly available.
- Made the Aseprite file parsing API publicly available.

### Fixed
- Fixed an issue where the Animation Window would no longer detect Animation Clips on a prefab. (DANB-458)

## [1.0.0-pre.4] - 2023-04-16
### Added
- Added a property to set the padding within each generated SpriteRect.
- Added an option to select import mode for the file, either Animated Sprite or Sprite Sheet.

### Fixed
- Fixed an issue where the platform settings could not be modified. (DANB-445)
- Fixed an issue where the Animation Events would be generated with the wrong time stamp.

## [1.0.0-pre.3] - 2023-03-23
### Added
- Burst compiled the texture generation tasks to speed up importation of Aseprite files. (Note: Only for Unity 2022.2 and newer).
- Layer blend modes are now supported with Import Mode: Merge Frames.
- Added ability to generate Animation Events from Cell user data.
- Added ability to export Animator Controller and/or Animation Clips.
- Added canvasSize to the Aseprite Importer's public API.

### Fixed
- Fixed an issue where the last frame in a generated Animation Clip would receive an incorrect length. (DANB-434)
- Improved the background importer, so that it only imports modified Aseprite files in the background.

## [1.0.0-pre.2] - 2023-02-27
### Added
- Added support for individual frame timings in animation clips.
- Added support for layer groups.
- Added support for Layer & Cel opacity.
- Added support for repeating/non-repeating tags/clips.

### Changed
- The importer UI is now re-written in UI Toolkit.
- If a Model Prefab only contains one SpriteRenderer, all components will be placed on the root GameObject, rather than generating a single GameObject to house them.
- A Sorting Group component is added by default to Model Prefabs with more than one Sprite Renderer.

### Fixed
- Fixed an issue where renaming an Aseprite file in Unity would throw a null reference exception. (DANB-384)
- Fixed an issue where the background importer would import assets even when Unity Editor has focus.
- Fixed an issue where the Pixels Per Unit value could be set to invalid values.

## [1.0.0-pre.1] - 2023-01-06
### Added
- First release of this package.
