# Changelog

## [10.1.0] - 2025-03-07
### Changed
- Update minimum Unity version.

## [10.0.7] - 2024-10-07
### Fixed
- DANB-731 "System.IndexOutOfRangeException" is thrown when increasing Sprite Shape Mesh size beyond limits

## [10.0.6] - 2024-05-22
### Changed
- Update Spriteshape package to 2D Common package 9.0.5 dependency.

## [10.0.5] - 2024-05-06
### Fixed
- DANB-604 Fix case where Spriteshape vertex array exceeds limit even though it has not reached 64K.

## [10.0.4] - 2024-04-01
### Changed
- Updated the Editor Analytics to use the latest APIs.

## [10.0.3] - 2024-02-06
### Fixed
- DANB-316 Update obsolete API usage.
- DANB-554 Prevent Crash on DynamicHeapAllocator::Deallocate when changing SpriteShapeController splineDetail value through Script to 1
- DANB-545 Fix rendering artefacts when using Variant Atlas for SpriteShape

## [10.0.2] - 2023-08-15
### Fixed
- UUM-41920 Fill tessellation doesn't work and Warnings logged for Closed 2D Shapes when C# Job is used

## [10.0.1] - 2023-04-28
### Fixed
- UUM-29635 Fix case where SpriteShape point overlay is displaying inconsistent number of digits after decimal point.
- DANB-422 Fix case where Sprite Shape is not filled when "Closed Sprite Shape" is enabled and "Detail" is set to "High Quality"
- DANB-403 Fix case where Spriteshape does not conform to edge sprites 9-slicing when first point is continuous.

## [10.0.0] - 2023-02-23
### Changed
- Release for Unity 2023.1

## [10.0.0-pre.2] - 2022-11-30
### Added
- Added Sample for Custom Geometry Generation and Vertex Colors.
- Replace obsolete API calls.

### Fixed
- DANB-243 Fix case where Sprite Shape is invisible when spawned at runtime
- DANB-183 Fix case where Spriteshape with tangents enabled has shadow artifacts after reopening the project
- DANB-282 Fix Case where BezierUtility.BezierPoint function parameters are not in the right order

## [10.0.0-pre.1] - 2022-09-21
### Fixed
- DANB-18 Fix case where SpriteShapeController does not initialize collider data.
- DANB-97 Fix SpriteShape Extras "Colliders" scene causes warnings in console when opening
- DANB-120 Fix case where adding a new element to Custom Geometry Modifier spams errors.

### Changed
- Refactored internal triangulation and tessellation APIs.
- D2D-3550 Move "Pixel Per Unit" and "Worldspace UV" fill settings adjacent to "Stretch UV" to have all fill settings grouped together

## [9.0.0-pre.1] - 2022-03-21
### Added
- Added versioning for GeometryCreator and GeometryModifier scripts so SpriteShape geoemetry is regenerated when it changes.
- Fill Tessellation in C# Job is now set as default tessellator. GC allocations are reduced when using this option.

### Fixed
- 1394404 Fix case where Tangent Data is always saved even when not in use for SpriteShapeRenderer when GeometryCache is active.
- 1391968 Fix case where "Invalid memory pointer was detected in ThreadsafeLinearAllocator::Deallocate!" error is thrown when Sprite is in Atlas with Tight Packing
- 1399392 Fix case where SpriteShape with Cache Geometry enabled does not update arrays when saving scene off-screen.
- 1400229 Fix case where SpriteShape corner does not respect the ControlPoint height.
- 1387298 Fix case where SpriteShape throws ArgumentException error when checking position validity of a point added to Spline
- 1401376 Fix case where Shape of PolygonCollider2D doesn't update when Sprite Shape contains vertex at [0,0] and it's Profile doesn't have any Sprites.

## [8.0.0] - 2022-01-25
### Changed
- Package release version.

### Fixed
- 1392653 Fix case where SpriteShapeGeometry Cache does not update when n selecting a different Object when EditTool is active.

## [8.0.0-pre.5] - 2021-11-24
### Fixed
- 1367509 Fix case where SpriteShapeProfile corner fields disappear when a field above has been deleted in the Inspector.
- 1363468 Fix case where shortcut keys do not work after editing sprite variant list in SpriteShape Controller.
- 1382718 Fix Case when setting SpriteShape corners to Disabled, the first corner does not visually change

## [8.0.0-pre.4] - 2021-10-21
### Changed
- Update to latest com.unity.2d.common package

## [8.0.0-pre.3] - 2021-10-18
### Fixed
- Fixed package.json to remove com.unity.2d.path dependency.

## [8.0.0-pre.2] - 2021-10-11
### Fixed
- 1368107 Fix case where Bounds can cause spriteshape not load in when running Player.
- 1364012 Fix crash when optimizing the geometry of the SpriteShape in certain cases.

## [8.0.0-pre.1] - 2021-08-06
### Added
- Add actionable console log when encounter vertex count limit exception
- Improved estimation of vertices required for geometry to minimize memory alloction.
- Added user preferences for Controlpoint/Tangent/Spline color.
- Added support for Global Grid Snapping. 

### Changed
- Remove dependency to path package
- Moved Control point specific data from Inspector to a Scene Overlay Window.

### Fixed
- Fixed Bounds of SpriteShapeRenderer.
- Update manual to reflect reorganization of menu item.
- 1346430 Fix case where all open Scenes were dirtied when editing a SpriteShape Spline.
- 1343836 Fix case where triangular spriteshape with 0 offset collider does not generate collision shape.
- 1356204 Fix case where Sprite Shapes appear only when their pivot is revealed in the Scene view.
- 1348701 Fix case where colliders do not extend to the end of the sprite texture when Sprite Borders are enabled.
- 1362440 Fix case where Edge and Polygon colliders have missing edges on certain open-ended shapes.
- 1363215 Fix case where enabling Fill Tessellation and setting profile's fill offset to positive causes errors.
- 1368129 Fix case where Sprite Shape default materials were not initialized correctly.


## [7.0.0-pre.3] - 2021-05-17
### Changed
- Update dependency version

## [7.0.0-pre.2] - 2021-05-14
### Changed
- Update dependency version

## [7.0.0-pre.1] - 2021-05-05
### Fixed
- 1274010 2D light is rendered in half in its Y-axis when two Sprite Shape objects with same Order In Layer are visible on the Screen
- 1313579 SpriteShape Prefabs does not work properly when GeometryCache is enabled.
- 1315086 When SpriteShapeController has "Update Collider" set to true, it will dirty the scene every time its selected
- 1306434 PrefabStage is moving out of UnityEditor.SceneManagement.Experimental namepace in 2021.2
- 1319096 At certain cases, vertex data allocation may not be enough and overflows.
- 1321978 Edge collider 2D and polygon collider 2D creates different collision shapes during playmode
- 1317728 On deselecting game object from the Inspector window leads to deselecting Sprite Shape Renderer
- 1326983 SpriteShape Cache Geometry does not update when changing SpriteShape Profile.

### Changed
- Version bump for Unity 2021.2

## [6.0.0] - 2021-03-17
### Changed
- Update version for release

## [6.0.0-pre.3] - 2021-02-28
### Fixed
- 1294930 Exception thrown continuously on creating Range in the preset of Sprite Shape when Undo/Redo operation is performed earlier
- 1303998 Enabling Fill Tessellation on controller and setting the profile's fill offset to negative causes errors
- 1293760 Sprite Shape generates Edge Colliders with deformed corners     
- 1305867 Sprite shape edge collider has a gap at end point if optimise collider is disabled
- 1286378 Sprite Shape incorrect normal generation

## [6.0.0-pre.2] - 2020-11-25
### Changed
- Update license file

### Fixed
- 1273635 Fixed error when adding AngleRange to SpriteShapeProfile Preset that was reset before.
- 1287237 Fixed ArgumentException when tangents and cache geometry are enabled on SpriteShapeController component.
- 1240514 Fixed InvalidOperationException thrown continuously on adding SpriteShapeController component to a GameObject with SpriteRenderer.
- 1284920 Fixed PolygonCollider2D generated with a single vertex when a GameObject has a SpriteShapeController with just 3 vertices.

## [6.0.0-pre.1] - 2020-10-30
### Changed
- Version bump for Unity 2021.1
- Height is interpolated linearly between control points that are both linear and smoothly if otherwise.  

## [5.1.0] - 2020-09-24
### Added
- Added C# Job Tessellation support for Fill Area of SpriteShape.

### Fixed
- 1274400 SpriteShape Bounding Box does not take into account certain vertices
- 1273705 Assertion failed exception is thrown on undoing after clicking on Create Range button
- 1273635 Errors occurs when adding range on Reset-ed Preset of the SpriteShapeProfile
- 1271817 Icon is missing on creating SpriteShapeProfile at the time of creating
- 1280016 Unable to create Sprite Shape Profile along with ArgumentNullException thrown in the Project window
- 1274776 NullReferenceException thrown on performing Redo operation after creating Range property in the SpriteShape profiler preset

## [5.0.2] - 2020-08-31
### Fixed
- 1267542 Sprite Variant Window does not appear in Sprite Shape Controller Component when selecting a Spline pivot point.
- 1265846 Dragging Sprite Shape Profile to Hierarchy creates a Game Object in main Scene when being in Prefab Mode

## [5.0.1] - 2020-07-17
### Changed
- If Geometry is baked using SpriteShapeGeometryCache, do not check for dirty once data is updated to prevent GC.
- Updated Help Section to point to the correct URLs.

### Fixed
- 1242910 Unable to add item on Resetting the Preset of the SpriteShapeProfile
- 1256914 Exception thrown continuously when Undo operation is performed with sprites are assigned earlier
- 1263266 BakeCollider requires GC every frame even when there are no changes in SpriteShape

## [5.0.0] - 2020-05-28
### Added
- Sample script GenerateSpriteShapes.cs to demonstrate force generating invisible SpriteShapes on runtime scene load.

### Changed
- Version bump for Unity 2020.2

### Fixed
- 1246133 Error occurs when unselecting Cache Geometry for Sprite Shape prefab
- 1240380 OnGUI in SpriteShapeController creates GC allocs.
- 1235972 "A Native Collection has not been disposed, resulting in a memory leak" is thrown when 2D Sprite Shape Controller is disabled
- 1240514 InvalidOperationException thrown continuously on adding "Sprite Shape Controller" Component to a Sprite object
- 1241841 Disabled corner option does not work on existing spriteshape upgraded from a previous version

## [4.1.1] - 2020-04-20
### Added
- Added BakeMesh to save generated geometry data.
- Added warning when a valid SpriteShape profile is not set.

## [4.1.0] - 2020-03-16
### Added
- Stretched Corners.

### Fixed
- 1226841 Fix when Collider generation allocation.
- 1226856 SpriteShape Edge Collider does not extend to End-point even if Edges dont overlap.
- 1226847 SpriteShape Corner Threshold does not work.


## [4.0.3] - 2020-03-09
### Fixed
- 1220091 SpriteShapeController leaks memory when zero control points are used
- 1216990 Colliders should also respect Pivot property of Edge Sprites.
- 1225366 Ensure SpriteShape are generated when not in view on Runtime.

## [4.0.2] - 2020-02-11
### Changed
- Improved Memory Allocations.

### Fixed
- Fixed OnDrawGizmos to Get/Release RenderTexture through CommandBuffer.

## [4.0.1] - 2019-11-26
### Changed
- Updated License file
- Updated Third Party Notices file
- Changed how Samples are installed into user's project

### Fixed
- Fixed where the last point of the Sprite Shape does not behave correctly when using Continuous Points in a closed shape (case 1184721)

## [4.0.0] - 2019-11-06
### Changed
- Update version number for Unity 2020.1

## [3.0.7] - 2019-10-27
### Fixed
- Added missing meta file

### Changed
- Update com.unity.2d.path package dependency

## [3.0.6] - 2019-09-27
### Added
- Added support to set CornerAngleThreshold.
- Burst is now enabled for performance boost.
### Fixed
- Fix (Case 1041062) Inputting Point Position manually causes mesh to not conform to the spline
- Fix GC in confirming Spline Extras sample.
- Fix hash Validation errors.
- Removed resources from Packages.

## [3.0.5] - 2019-09-05
### Fixed
- Fix (Case 1159767) Error generated when using a default sprite for Corner sprite or Angle Range sprite in Sprite Shape Profile
- Fix (Case 1178579) "ArgumentOutofRangeException" is thrown and SpriteShapeProfile freezes on reset

## [3.0.4] - 2019-08-09
### Added
- Added tangent channel support for proper 2D lighting in URP.

## [3.0.3] - 2019-07-24
### Added
- Add related test packages

## [3.0.2] - 2019-07-13
### Changed
- Update to latest Mathematics package version

## [3.0.1] - 2019-07-13
### Changed
- Mark package to support Unity 2019.3.0a10 onwards.

## [3.0.0] - 2019-06-19
### Changed
- Stable Version.
- Remove experimental namespace.

## [2.1.0-preview.8] - 2019-06-12
### Changed
- Fix (Case 1152342) The first point of the Sprite Shape does not behave correctly when using Continuous Points
- Fix (Case 1160009) Edge and Polygon Collider does not seem to follow the spriteshape for some broken mirrored tangent points
- Fix (Case 1157201) Edge Sprite Material changed when using a fill texture that is already an edge sprite on spriteshape
- Fix (Case 1162134) Open ended Spriteshape renders the fill texture instead of the range sprite

## [2.1.0-preview.7] - 2019-06-02
### Changed
- Fix Variant Selection.

## [2.1.0-preview.6] - 2019-06-02
### Changed
- Fix Null reference exception caused by SplineEditorCache changes.
- Fill Inspector changes due to Path integration.

## [2.1.0-preview.4] - 2019-05-28
### Changed
- Upgrade Mathematics package.
- Use path editor.

## [2.1.0-preview.2] - 2019-05-13
### Changed
- Initial version for 2019.2
- Update for common package.

## [2.0.0-preview.8] - 2019-05-16
### Fixed
- Fixed issue when sprites are re-ordered in Angle Range.
- Updated Samples.

## [2.0.0-preview.7] - 2019-05-10
### Fixed
- Version Update and fixes.

## [2.0.0-preview.6] - 2019-05-08
### Fixed
- Added Sprite Variant Selector.
- Fix Variant Bug (https://forum.unity.com/threads/spriteshape-preview-package.522575/page-6#post-4480936)
- Fix (Case 1146747) SpriteShape generating significant GC allocations every frame (OnWillRenderObject)

## [2.0.0-preview.5] - 2019-04-18
### Fixed
- Shape angle does not show the accurate sprite on certain parts of the shape.
- SpriteShape - Unable to use the Depth buffer (https://forum.unity.com/threads/spriteshape-preview-package.522575/page-6#post-4413142)

## [2.0.0-preview.4] - 2019-03-28
### Changed
- Disable burst for now until we have a final release.

## [2.0.0-preview.3] - 2019-03-25
### Fixed
- Update Common version.

## [2.0.0-preview.2] - 2019-03-08
### Fixed
- Fix Edge Case Scenario where Vertices along Continuous segment could be duplicated..
- Ensure that Collider uses a valid Sprite on Generation.

## [2.0.0-preview.1] - 2019-02-27
### Changed
- Updated version.

## [1.1.0-preview.1] - 2019-02-10
### Added
- Spriteshape tessellation code is re-implemented in C# Jobs and utilizes Burst for Performance.
- Added Mirrored and Non-Mirrored continous Tangent mode.
- Simplified Collider Generation support and is part of C# Job/Burst for performance.
- Added Shortcut Keys (for setting Tangentmode, Sprite Variant and Mirror Tangent).
- Ability to drag Spriteshape Profile form Project view to Hierarchy to create Sprite Shape in Scene.
- Simplified Corner mode for Points and is now enabled by default.
- Added Stretch UV support for Fill Area.
- Added Color property to SpriteShapeRenderer.

### Fixed
- SpriteShapeController shows wrong Sprites after deleting a sprite from the top angle range.
- Empty SpriteShapeController still seem to show the previous Spriteshape drawcalls
- Streched Sprites are generated in between non Linked Points
- Corners sprites are no longer usable if user only sets the corners for the bottom
- Sprites in SpriteShape still shows even after user deletes the SpriteShape Profile
- SpriteShape doesn't update Point Positions visually at runtime for Builds
- Spriteshape Colliders does not update in scene immediately
- Fixed constant Mesh baking (https://forum.unity.com/threads/spriteshape-preview-package.522575/page-4#post-3925789)
- Fixed Bounds generation issue (https://forum.unity.com/threads/spriteshape-preview-package.522575/page-5#post-4079857)
- Sprite Shape Profile component breaks when creating range
- Fixed when sprite is updated in the sprite editor, the spriteshape is not updated.
- Fixed cases where Spline Edit is disabled even when points are selected. (https://forum.unity.com/threads/spriteshape-preview-package.522575/#post-3436940)
- Sprite with SpriteShapeBody Shader gets graphical artifacts when rotating the camera.
- When multiple SpriteShapes are selected, Edit Spline button is now disabled. (https://forum.unity.com/threads/spriteshape-preview-package.522575/page-3#post-3764413)
- Fixed texelSize property (https://forum.unity.com/threads/spriteshape-preview-package.522575/page-4#post-3877081)
- Fixed Collider generation for different quality levels. (https://forum.unity.com/threads/spriteshape-preview-package.522575/page-4#post-3956062)
- Fixed Framing Issues (https://forum.unity.com/threads/spriteshape-preview-package.522575/page-5#post-4137214)
- Fixed Collider generation for Offsets (https://forum.unity.com/threads/spriteshape-preview-package.522575/page-5#post-4149841)
- Fixed Collider generation for different Heights (https://forum.unity.com/threads/spriteshape-preview-package.522575/page-5#post-4190116)

### Changed
- SpriteShape Asset parameters WorldSpace UV, PixelPerUnit have been moved to SpriteShapeController properties.
- Collider generation has been simplified and aligns well with the generated geometry (different height, corners etc.)

### Removed
- Remove redundant parameters BevelCutoff and BevelSize that can be done by simply modifying source spline.

## [1.0.12-preview.1] - 2018-08-03
### Added
- Fix issue where Point Positions do not update visually at runtime for Builds

## [1.0.11-preview] - 2018-06-20
### Added
- Fix Spriteshape does not update when Sprites are reimported.
- Fix SpriteShapeController in Scene view shows a different sprite when user reapplies a Sprite import settings
- Fix Editor Crashed when user adjusts the "Bevel Cutoff" value
- Fix Crash when changing Spline Control Points for a Sprite Shape Controller in debug Inspector
- Fix SpriteShape generation when End-points are Broken.
- Fix cases where the UV continuity is broken even when the Control point is continous.

## [1.0.10-preview] - 2018-04-12
### Added
- Version number format changed to -preview

## [0.1.0] - 2017-11-20
### Added
-  Bezier Spline Shape
-  Corner Sprites
-  Edge variations
-  Point scale
-  SpriteShapeRenderer with support for masking
-  Auto update collision shape
