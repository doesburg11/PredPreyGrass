# Changelog

## [9.1.0] - 2025-03-07
### Changed
- Update minimum Unity version.

### Fixed
- DANB-811 Fix blurry Sprite Atlas Sample variant scene sprites
- DANB-794 Fixed case where editing the Closed and Open Sprite Shapes is choppy

## [9.0.7] - 2024-10-07
### Fixed
- DANB-638 Fixed Error "InvalidOperationException: HTTP/1.1 404 Not Found" logged when entering Play Mode in 2D Common Sample Scene
- DANB-637 Fixed Sprite Atlases included in the 2D Common Package Sample "Sprite Atlas Samples" are blurry even though they are uncompressed

## [9.0.6] - 2024-07-26
### Added
- Internal functionality to support 2D Muse.
- Internal support for AssetPreview methods.

### Changed
- Atlas sample to use correct filepath when building for Android.

## [9.0.5] - 2024-05-16
- ### Fixed
- DANB-604 Fix case where Spriteshape vertex array exceeds limit even though it has not reached 64K.

## [9.0.4] - 2024-02-06

### Changed
- Internal changed to use public shortcut API.

## [9.0.3] - 2023-11-30
### Added
- Add ability to add lock to EditorWindow

## [9.0.2] - 2023-08-15
### Added
- Added Samples for SpriteAtlas.

### Fixed
- Fixed case where PVRTC compression format for the iOS platform is not supported. (Case DANB-500)

(Case DANB-500)

## [9.0.1] - 2023-05-03
### Fixed
- Fixes for internal APIs.

## [9.0.0] - 2023-02-23
### Changed
- Release for Unity 2023.1

## [9.0.0-pre.2] - 2022-11-30
### Added
- Added mipmap streaming data to the mipmap settings class.
- Added Scriptable Packer Object that can be used in a Sprite-Atlas for Optimal packing of Sprites.

## [9.0.0-pre.1] - 2022-09-21
### Changed
- Refactored internal triangulation and tessellation APIs.
- Update com.unity.burst dependency version to 1.7.3 to support latest PS4 SDK.
- Mark package for Unity Editor 2023.1.
- Added support for different sized texture inputs in ImagePacker. 

## [8.0.0-pre.2] - 2022-05-31
### Added
- Moved internal API from animation to common package.

## [8.0.0-pre.1] - 2022-03-21
### Changed
- Minimized memory allocated for UTess.

## [7.0.0] - 2022-01-25
### Changed
- Package release version.

### Fixed
- 1382695 Fixed case where control point selection flickers when drag and multi-select points in scene
- Optimized texture space needed for rect packing

## [7.0.0-pre.4] - 2021-11-24
### Added
- Added internal method to get build target's group name.
- Added access to the internal helper method IsUsingDeformableBuffer.

### Fixed
- Allow internal TextureGenerator helper consider swizzle data. 

### Fixed
- 1368956 Deleting certain vertices in sprite mesh leads to mesh resetted to quad incorrectly

## [7.0.0-pre.3] - 2021-10-21
### Fixed
- Fixed passing in invalid argument to TextureGenerator for swizzling.

## [7.0.0-pre.2] - 2021-10-11
### Fixed
- 1361541 Fix crash encountered when deleting vertices of sprite mesh in SkinningEditor

## [7.0.0-pre.1] - 2021-08-06
### Changed
- Update Unity supported version to 2022.1

## [6.0.0-pre.4] - 2021-07-05
### Added
- Internal API for applying Sprite Editor Window changes

## [6.0.0-pre.3] - 2021-05-19
### Fixed
- Fixed issues in tesselation library.

## [6.0.0-pre.2] - 2021-05-14
### Fixed
- Fixed metafiles conflicts

## [6.0.0-pre.1] - 2021-05-05
### Changed
- Version bump for Unity 2021.2

## [5.0.0] - 2021-03-17
### Changed
- Update version for release

## [5.0.0-pre.2] - 2021-01-16
### Changed
- Update license file

## [5.0.0-pre.1] - 2020-10-30
### Changed
- Version bump for Unity 2021.1

## [4.0.3] - 2020-10-15
### Fixed
- Allow 2D Packages to access internal constant value for asset creation instance id

## [4.0.2] - 2020-08-31
### Fixed
- Allow launching Sprite Editor Window to target a specific asset

## [4.0.1] - 2020-07-07
### Fixed
- Updated to use non-experimental AssetImporter namespace (case 1254381)

## [4.0.0] - 2020-05-11
### Changed
- Version bump for Unity 2020.2

## [3.0.0] - 2019-11-06
### Changed
- Update version number for Unity 2020.1

## [2.0.2] - 2019-08-09
### Added
- Add Seconday Texture settings into TextureSettings for TextureGenerator
- Add related test packages

## [2.0.1] - 2019-07-13
### Changed
- Mark package to support Unity 2019.3.0a10 onwards.

## [2.0.0] - 2019-06-17
### Added
- Drop preview tag.
- Remove experimental namespace

## [1.2.0-preview.2] - 2019-06-04
### Added
- Remove Image Packer Debug Window
- Move tests out of package

## [1.2.0-preview.1] - 2019-02-20
### Added
- Update for Unity 2019.2 support.

## [1.1.0-preview.2] - 2019-03-18
### Added
- Remove deprecated call to Unity internal API

## [1.1.0-preview.1] - 2019-01-25
### Added
- Added versioning for CI.