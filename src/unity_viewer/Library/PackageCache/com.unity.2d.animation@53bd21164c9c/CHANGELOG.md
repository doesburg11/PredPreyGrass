# Changelog

## [10.2.0] - 2025-03-07
### Changed
- Update minimum Unity version.

### Fixed
- Sprite Skin's test instability. (DANB-772) 
- Sprite Skin's null ref exception when HDRP is present in the project. (DANB-785)
- Updating Skinning Plugin to work on Ubuntu versions >= 20.
- Skinning Editor style fixes. (DANB-804)
- Reset TransformAccessJob's cache. (DANB-807)
- Asset Upgrader icons. (DANB-802)

## [10.1.4] - 2024-10-07
### Fixed
- Animation Preview window sometimes does not display deformed Sprites. (DANB-705)
- Sprite Resolver missing sprite previews when dealing with large number of entries. (DANB-714)
- Misaligned label previews in Sprite Resolver's inspector. (DANB-722)
- Sprite Resolver component not updated after Sprite Library Asset has been modified. (DANB-727)
- Sprite Skin breaks in the animation preview window after sprite swap. (DANB-743)
- IK gizmos are displayed in the SceneView when IKManager2D is active in Animation Preview window. (DANB-738)
- IK solvers are misaligned when bones have different depths. (DANB-753)
- Rendering issues with SRP Batching and Sprite mask. (DANB-760)
- Unable to drag sprites into empty rows of the Sprite Library Editor. (DANB-749)
- Sprite Skin deformation systems have outdated data after sprite swap. (DANB-766)
- Setting incorrect computer buffer size. (DANB-768)
- Reenable editor tests.
- Bone buffer binding issues.
- Sprite changed callback listeners.

## [10.1.3] - 2024-07-26
### Fixed
- Sprite Swap overlay Label selection not reacting to mouse clicks. (case DANB-627)
- Sprite Swap overlay selection lock behavior. (case DANB-629)
- Sprite Bone Influence tab bone reordering. (case DANB-634)
- Documentation, comments, and other requirements to conform to Package Validation. (DANB-653)
- Wrong Label is deleted when a newly created Label is deleted in the Sprite Library Editor. (case DANB-665)
- Cannot select multiple Labels in the Sprite Library Editor when using Grid View. (case DANB-670)
- Documentation, comments, and other requirements to conform to Package Validation. (DANB-653)
- Support for renaming Sprite Library Editor entries with 'F2' on Windows. (DANB-674)
- Renaming functionality for Labels is available to users when multiple Labels are selected. (case DANB-668)
- Fixed SpriteSkin's playmode test failures. (case DANB-678)
- Modifying the Sprite Skin component through public API.
- Skinning Editor bone and mesh Visibility sliders too short. (case DANB-681)
- Fixed an issue where opening a Sprite Library Asset, when it is a sub asset, in the Sprite Library Editor, would throw an exception.

## [10.1.2] - 2024-05-06
### Fixed
- Sprite Library's labels and categories cannot be reordered. (case DANB-593)
- Added safety checks to SpriteSkin's public APIs. (case DANB-598)
- Fixed the IK bones flipped at certain angels. (case DANB-595)
- Fixed the the memory leak in the SpriteSkin component. (case DANB-603)
- Fixed the incorrect transform cache capacity in the SpriteSkin component. (case DANB-615)

### Changed
- Updated the Editor Analytics to use the latest APIs.
- Updated the UI Elements to use the latest APIs.
- Remove implicit NativeArray cast from MeshUtilities.
- Updated the SpriteSkin's deformation system to improve performance.

## [10.1.1] - 2024-02-06
### Added
- Support for automatic weight generation in the Skinning Editor for ARM64 editors. (case DANB-550)

### Fixed
- Sprite Resolver previews not visible after editor regained focus. (case DANB-565)
- Sprite Skin is not deformed in the Editor after exiting the playmode if the domain reload is disabled. (case DANB-573)

## [10.1.0] - 2023-11-30


### Fixed
- Blend weight vertex attribute is added only when sprite has bones. (case DANB-521)
- Sprite Library editor drag and drop interactions. (case DANB-536)
- Sprite Library editor drag and drop blocked by labels. (case DANB-560)
- Adjust the copy-paste logic to not paste sprite mesh to the same sprite twice. (case DANB-532)
- Sprite Library Asset is selected on creation. (case DANB-544)
- Sprite Library Editor window not reacting to the delete command. (case DANB-547)

### Added
- SpriteSkin's Auto Rebind property can now be accessed from scripts. (case DANB-492)
- Sprite Library public API to save an asset as a .spriteLib SpriteLibrarySourceAsset.
- Ability to lock Sprite Library Editor window's asset selection.

### Changed
- Update samples to use UI Toolkit.
- Moved the Sprite Library Editor window's save shortcut to the Shortcut Manager.

## [10.0.3] - 2023-09-04

### Fixed
- Fixed an issue where the FK doesn't blend correctly with the IK if solver solves from the bind pose. (case DANB-484)
- Fixed an issue where GPU-deformed Sprites will not animated after scene reload. (case DANB-461)
- Added correct help documentation link to Sprite Library Asset. (case DANB-487)

### Changed
- Updated documentation on enabling GPU skinning.
- Update the Sprite Library component inspector to inform users about the entry overrides documentation page.

## [10.0.2] - 2023-05-04
### Added
- Updated IK parameter names and comments in public APIs.

### Fixed
- Fixed Sprite Resolver inspector not able to set category to 'No Category' if it contains the old hash value. (case DANB-396)
- Fixed an issue where the Limb IK solver will become unstable when the child bones have different Z position. (case DANB-416)
- Fixed an issue where the CCD and Fabrik solvers will become unstable when the child bones have different Z position. (case DANB-421)
- Fixed an issue where changing Sprite Library reference in the Inspector will result in an exception. (case DANB-429)

## [10.0.1] - 2023-05-03
### Added
- Added console log when Skinning Editor is overwriting mesh from Custom Outline module. Logs can be enabled or disabled from 2D Animation user settings.
- Added support for the Collections 2.X packages.

### Fixed
- Fixed Sprite Resolver inspector not able to set category to 'No Category' if it contains the old hash value. (case DANB-396)


### Changed
- Renamed the "Sprite Resolver" overlay to "Sprite Swap".
- Added lock toggle to the Sprite Swap overlay and updated its stylesheets.
- Added support for the Collections 2.X packages.

## [10.0.0] - 2023-02-23

### Added
- Scene overlays for manipulating Sprite Resolver's category and label.
- Ability to change Sprite Skin's root transform and bone transforms from a script.
- Ability to reset Sprite Skin's bind pose by script.

### Fixed
- Fixed an issue where undoing vertex painting results in a displaced mesh. (case DANB-334)
- Fixed an issue where "System.ArgumentException" thrown when opening sprite Prefab with Script. (DANB-335)
- Fixed an issue where IKEditorManager updates even when there are no active IK Manager 2Ds in the Scene. (case DANB-348)
- Fixed SpriteResolver does not animation with animation clip when upgrading from 2020.3. (case DANB-375)
- Fixed an issue where IKGizmos will throw an exception in the editor when IKSolver has an unassigned Solver in its list. (case DANB-373)
- Fixed IMGUIContainer:ProcessEvent error occurs when selecting a PSD embedded SpriteLibraryAsset as a Main Library of another Sprite Library Asset. (case DANB-379)
- Fixed Sprite Resolver inspector not updating its previews when the category is set externally to 'No Category'. (case DANB-390)


## [10.0.0-pre.2] - 2022-11-30
### Fixed
- Fixed an issue where body parts could not be selected from UI dropdowns in a sample scene '2 Part Swap'. (case DANB-262)
- Fixed case where IndexOutOfRangeException occurs when auto-generating mesh with a specific settings and bones. (Case DANB-244)
- Split the batch deformation into two, GPU & CPU deformation, which allows renderers to be placed into the deformation system most suitable for the renderer.
- Optimized the bone gizmo rendering in the scene view.
- Replace obsolete API calls.

### Changed
- Changed the default script execution order for of Sprite Resolvers, IK Managers and Sprite Skins.

### Added
- Sprite Library Assets can now be dragged and dropped into the scene, hierarchy, and inspector.

## [10.0.0-pre.1] - 2022-09-21
### Added
- Added bone weight index validation in SpriteSkin's validate method, to ensure valid data before continuing with deformation.
- IK Manger 2D now supports camera frustum culling.
- Added internal properties to Sprite Skin to get outline data of an assigned mesh.

### Fixed
- Fixed a case where moving vertices forcefully in the Skinning editor could cause a quad reset of the mesh. (case DANB-7)
- Fixed a case where pasting bones in the Skinning Editor would move bones rather than copy them. (case DANB-99)
- Fixed an issue where selecting bones in the Skinning Editor after removing any bone in the skeleton will throw an exception. (Case DANB-124)
- Fixed a case where setting IKManager2D's or Solver2D's weight to '0' doesn't update solver's effector position. (DANB-139)
- Fixed a case where multi selecting Sprite Skins would cause a null reference exception to be thrown. (case DANB-126)
- Fixed an issue where undo the addition of a Sprite Skin component would crash the editor. (DANB-201)
- Fixed an issue where the Sprite Skin editor would throw an exception if Sprite Renderer doesn't have a Sprite assigned to it.
- Fixed a case where new bones are not selected after pasting them in the Skinning Editor and an exception is thrown. (DANB-208)
- Fixed IK Manager 2D's inspector slow downs. (case DANB-214)
- Fixed a case where a .psd/.psb with a Main Skeleton assigned would generate incorrect bind poses. (Case DANB-225)

### Changed
- Refactored internal triangulation and tessellation APIs.
- Disabled Sprite Library modification in the Component's Inspector and added a button to export changes to an Asset.
- Pasting unassociated data to Skinning Editor doesn't throw console errors.
- Expand and frame on bone selection.

## [9.0.0-pre.3] - 2022-05-31
### Changed
- Update dependency package version.

## [9.0.0-pre.2] - 2022-05-20
### Fixed
- Creating a new vertex or an edge outside of the main geometry is now handled correctly. (case 1398541)
- Fixed an issue where null reference exceptions would show when the Skinning Editor was open during play mode. (case 1419720)
- Fixed Sprite Library Asset 'Revert' behavior. (case 1417743)
- Fixed Sprite Library Asset multi-editing in the Inspector window. (case 1417747)
- Fixed an issue where Variant Sprite Libraries would not be updated when the Main Library is changed. (case DANB-5)

### Changed
- Added ability to create Sprite Library Asset variant from the create menu.
- Added dialog box to the Skinning Editor when entering Play Mode to allow saving unsaved changes.

## [9.0.0-pre.1] - 2022-03-21
### Added
- Added the ability to cancel mesh generation in the Skinning Editor.

### Changed
- 2D Animation now depends on the Collections package.
- Sprite Skin's AutoRebind can now swap between all bones underneath the rootBone.
- Updated Visibility Tab documentation page.
- Sprite Skins are now registered/deregistered in batches for improved instatiation/destroy performance.
- Updated ListView event listeners to use new selection API.

### Fixed
- Fixed an issue where the animation window's preview of IK targets would not be the same as in Play Mode. (case 1391590)
- Sprite Library cannot reference itself in the Main Library field or any asset that references it. (case 1401464)
- Fixed an issue where moving vertices in the Skinning Editor might result in invalid edges. (case 1386153)
- Fixed an issue where the SpriteSkin components would not get initialized on editor launch. (case 1401139)
- Fixed an issue when Skinning Editor will lose reference after exiting Play mode. (case 1405289)
- Fixed an issue where the Skinning Editor's copy/paste tool would fail if two bones shared the same name. (case 1405028)

## [8.0.0] - 2022-01-25
### Changed
- Package release version.

### Fixed
- Sprite Skin's help button now leads to the correct documentation page. (case 1383765)
- Fixed the isolate behavior in the Sprite tab in the Visibility panel. (case 1387184)

## [8.0.0-pre.4] - 2021-11-24
### Added
- Added support in the asset upgrading tool for animation clips authored across multiple Unity editor versions.
- Added a check to validate collinearity among vertices before tessellation.

### Changed
- Updated the 2D Animation Upgrader documentation section with new information regarding Animation Clip upgrading.

### Fixed
- Added additional fallback for when the bursted tessellation step fails. (case 1372686)
- Sprite selection now reacts only to the left mouse button. (case 1371567)
- Fixed an issue where a Sprite Skin outside of the camera frustum enters the frustum, which could cause an editor and player crash. (case 1377867)
- Slider labels in the Skinning Editor now register input for easier value tweaking. (case 1294945)

## [8.0.0-pre.3] - 2021-10-21
### Changed
- Update to latest com.unity.2d.common package

## [8.0.0-pre.2] - 2021-10-11
### Changed
- Sprite Library Asset are now named "New Sprite Library Asset.spriteLib" by default on creation.
- Updated Toolbar and Visibility tab buttons' selection color.
- Updated package documentation.

### Fixed
- Sprite selection actions now work with Undo. (case 1367257)
- Fixed an issue where removing an edge in the Skinning editor would result in the mesh falling back to a quad. (case 1365831)
- Quads are now generated at correct positions after removing all vertices from Sprite meshes. (case 1366633)

## [8.0.0-pre.1] - 2021-08-06
### Added
- Added an asset upgrading tool, which can upgrade older Sprite Library Assets and Animation Clips to the latest version.

### Changed
- SpriteResolver.SetCategoryAndLabel and SpriteResolver.ResolveSpriteToSpriteRenderer now returns a bool to signal if the methods managed to resolve the request or not.
- Orientation function being replaced with a simpler version.

### Fixed
- Thumbnails in Sprite Library Asset flicker when the Library Asset contains many Categories and Labels. (case 1333228)
- SpriteLibraryAsset Category & Label does not generate hash in Inspector. (case 1340587)
- SpriteResolver inspector selects first element when failing to resolve. (case 1340070)
- IKManager2D does not detect classes inheriting from Solver2D. (case 1343260)
- Skinning Editor tooltips updated.
- Bone and Sprite influence lists are displayed correctly. (case 1349041)
- Sprite Library Assets are now being cached properly. (case 1347339)
- Sprite outline in the Skinning Editor is now rendered based on Sprite's geometry. (case 1335586)
- Animation Preview windows can now show Sprite deformation, Sprite Swapping and IK controlled movement.
- Removed the need to set Broken and Constant tangent on each key when animating a SpriteResolver.
- Fixed render texture size error in the Skinning Editor. (case 1357552)
- Skinning Editor toolbar buttons now focus on hoykey presses. (case 1358714)
- Fixed an issue where opening certain .psb files would result in errors. (case 1358972)
- Fixed a case where quads generated in the Skinning Editor would be created with the wrong size and position. (case 1361053)
- Fixed an issue where variant Sprite Libraries would not display its main library content. (case 1362389)
- Fixed an issue where IK Solvers would not be updated when previewing an animation clip. (1354389)
- Fixed an issue where copying mesh and bone data from a .psb containing a single sprite would throw an exception. (case 1351543)
- Fixed an issue where an error would show up when destroying a Sprite Skin component while deep profling. (case 1364910)

## [7.0.0-pre.3] - 2021-07-05
### Fixed
- Thumbnails in Sprite Library Asset flicker when the Library Asset contains many Categories and Labels. (case 1333228)
- SpriteLibraryAsset Category & Label does not generate hash in Inspector. (case 1340587)
- SpriteResolver inspector selects first element when failing to resolve. (case 1340070)
- IKManager2D does not detect classes inheriting from Solver2D. (case 1343260)
- Skinning Editor tooltips updated.

## [7.0.0-pre.2] - 2021-05-19
### Fixed
- Fixed Sprite Resolver component not updated when new categories/labels are added into Sprite Library asset. (case 1321069)

## [7.0.0-pre.1] - 2021-05-05
### Changed
- Version bump for Unity 2021.2.
- Moved Copy & Paste Rig buttons in the Skinning Editor into their own Rig category.
- Moved Preview & Restore Pose button in the Skinning Editor into their own Pose category.
- Replaced Triangle.Net with our own tessellation solution.

### Added
- Added shortcuts at the back of the Skinning Editor buttons tooltips for better discoverability.
- Added a color picker for each bone in the Skinning Editor's Visiblity tab.
- Added the Sprite Influence panel that allows to change a bone influence on the selected Sprite.

### Fixed
- Fixed crash when disabling Sprite Skin when multithreaded rendering is enabled. (case 1296355)

## [6.0.0] - 2021-03-17
### Changed
- Update version for release.

## [6.0.0-pre.3] - 2021-03-15
### Changed
- Updated manual.

### Fixed
- Deleting bones from a skeleton referenced by another character sometimes throws IndexOutOfRangeException. (case 1304768)

## [6.0.0-pre.2] - 2021-01-16
### Changed
- Update license file.

### Added
- Initial documentation update for new features for 6.0.0.

### Fixed
- SpriteResolver resets to the current Sprite from SpriteRenderer if exist in Sprite Library.

## [6.0.0-pre.1] - 2020-11-02
### Changed
- Sprite Swap related features moved out of experimental namespace.
- Removed editing of Sprite Swap feature in Skinning Module.
- Updated Sprite Swap workflow focusing on Project Window and Inspector.
- Sprite Library Asset is created via AssetImporter.

### Added
- Sprite Library Asset supports variant.
- Sprite Library supports override during editing.
- Supports sharing of bone structures in Skinning Module.
- Added position, rotation and bone color editing in Skinning Module.

### Fixed
- Added missing tooltips in the Sprite Skin inspector. (case 1285255)

## [5.0.3] - 2020-10-15
### Fixed
- Fixed Sprite with no animation data is being processed during AssetPostProcessor.
- Fixed properties under the Sprite Library Asset overlapping in inspector. (case 1280017)
- Fixed vertical slider handle is not aligned and placed slightly to the right side in the Bone Influence window. (case 1260568)

## [5.0.2] - 2020-08-31
### Fixed
- Fixed Visibility window overlaps with weights and geometry window when Sprite Editor Window resizes. (case 1263353)
- Fixed 'Depth' column label gets clipped in Visibility Tool Window. (case 1257991)
- Fixed 'Invalid worldAABB' error message when repeatedly pressing Pack Preview button. (case 1270150)
- Fixed Null reference exception when changing values of a material while recording animation with Skinning Module enabled. (case 1267300)
- Improved memory and speed of Animation SpritePostProcess for large sprite count.

## [5.0.1] - 2020-07-24
### Fixed
- Fixed Skinning module flickers when creating in category in Visibility Window. (case 1244097)
- Fixed NullReferenceException when creating Preset for SpriteSkin component. (case 1254873)
- Updated optional dependency support for Collections to 0.9.0-preview.6 and Burst 1.3.3. (case 1255839)

## [5.0.0] - 2020-05-11
### Changed
- Version bump for Unity 2020.2.

### Added
- Combined 2D IK package with 2D Animation package.

### Fixed
- Remove unused XR dependency. (case 1249390)
- Fixed NullReferenceException when creating prefab with SpriteSkin component. (case 1245149)

## [4.2.4] - 2020-05-19
### Fixed
- Fixed compilation error by depending on latest 2D Common package.

## [4.2.3] - 2020-04-26
### Changed
- Improved performance by batching buffer submission when Collection and Burst package is installed

### Fixed
- Fix 'ArgumentException' when creating UXML Template Asset with 2D Animation package installed (case 1239680)

## [4.2.2] - 2020-03-26
### Fixed
- Fixed bone name field in weight slider does not display bone name (case 1226249)
- Fixed SpriteResolver's Inspector not updated when GameObject is disabled

### Changed
- Improved deformation performance when Collection and Burst package is installed

### Added
- Allow reordering of bone order in Bone Influence window. This is to allow fine tuning of bone order when shown in SpriteSkin's Inspector

## [4.2.1] - 2020-03-20
### Fixed
- Fixed inconsistent line ending

## [4.2.0] - 2020-03-11
### Added
- Add alwaysUpdate option to SpriteSkin to determine if SpriteSkin execution should occur even when the associated SpriteRenderer is culled
- Added message to inform user on dependent packages when viewing certain sample Scenes
- Added API to access deformed vertices from SpriteSkin

### Changed
- Improved SpriteSkinEditor UI
- Adjust length of popup and value fields for Weight Slider Window

### Removed
- Remove Bounds Gizmo from SpriteSkin
- Remove Reset Bounds button from SpriteSkinEditor

### Fixed
- Fixed Sprite asset used by SpriteSkin in Scene is being deleted
- Fixed broken documentation links in inspectors
- Fixed Sprite deformation not updated when GameObject is being enabled
- Fixed exception after reverting from creating new vertices and edges
- Fixed visual defect after undoing changes to Bone Transform properties in SpriteSkin's Inspector

## [4.1.1] - 2020-01-20
###Fixed
- Fix 2D Animation not working when reloading scene in runtime (case 1211100)

###Added
- Bone visibility persist after apply
- Sprite visibility persist after apply

###Changed
- Deformed Sprite's bounds are now calculated and bounds property is removed from SpriteSkin's inspector (case 1208712)

## [4.1.0] - 2019-12-18
### Changed
- Changed default shortcut key for "Animation/Create Vertex" from "Shift-D" to "Shift-J"
- Changed default shortcut key for "Animation/Weight Brush" from "Shift-C" to "Shift-N"

### Fixed
- Fix visual glitch when using SpriteSwap with Multi-threaded rendering (case 1203380)
- Fix bone name misaligned under Weight Slider Inspector when a name contains more than 26 letters (case 1200873)
- Fix bones not chained correctly when splitting bone in certain cases
- Fix 'Label' and 'Sprite' name overlaps with its input field when preset of "Sprite Library Asset" is created (case 1201061)
- Fix bone names can be empty (case 1200861)
- Fix bone gets created even though clicked on Visibility Panel (case 1200857)
- Fix NullReferenceException when using shortcut 'Shift+1' in certain cases (case 1200849)

### Added
- Expose SpriteSkin component to be accessible from scripts.

## [4.0.1] - 2019-11-26
### Changed
- Changed how Samples are imported into the user's project
- Updated Third Party Notices file

### Fixed
- Fix Animation Samples crashes when installing on certain machines (case 1185787)

## [4.0.0] - 2019-11-06
### Changed
- Version bump for Unity 2020.1
- Improve optional performance boost by installing Burst and Collections package. Currently tested with
  - com.unity.collections 0.1.1-preview
  - com.unity.burst 1.1.2

### Added
- Skinning Module now persists the following state after Apply or Revert is clicked in Sprite Editor Window
  - Current view mode i.e. Character or Spritesheet Mode
  - Sprite Selection
  - Bone Selection
  - Preview Pose
  - Vertex Selection
  - Visibililty Tool Active State
  - Weight Brush Settings

### Fixed
- Update to use new UIElements ListView API

## [3.0.7] - 2019-10-18
### Fixed
- Fix Search reset button is visible in Visibility tool when nothing is entered into the search field (case 1182627)
- Fix Sprite outline disappears when the Selected Outline Color alpha value is less than 255 (case 1186776)
- Fix button's label spelling error in 'Generate For All Visible' (case 1188621)

## [3.0.6] - 2019-09-18
### Changed
- Remove usage of Resource folder for assets needed by package.

### Fixed
- Fix GC allocation when using Sprite Resolver.

## [3.0.5] - 2019-09-06
### Added
- Optional performance boost by installing Burst package.
- Samples showing different how to produce different outcomes

### Changed
- Sprite and Group in Sprite Visibility Window appear in same order as original art source file

### Fixed
- Fix missing bone data in Sprite when importing with AssetDatabase V2

## [3.0.4] - 2019-08-09
### Added
- Add related test packages
- Added tangent deform for lighting support

### Fixed
- Fixed Amount slider not working in Weight Slider Panel
- Fixed exception when changing size to less than 0 in SpriteLibraryAssetInspector
- Fixed Sprite visual corruption when swapping Sprite using SpriteResolver

###Changed
- Make Size property field in Weight Brush draggable for changing brush size
- Rename SpriteLibraryAsset::GetCategorylabelNames to SpriteLibraryAsset::GetCategoryLabelNames
- Change string hash for Category and Label name. This might break existing animation usage with SpriteResolver.
- Add Experimental tag on Sprite Swap related features


## [3.0.3] - 2019-07-17
### Changed
- Update documentation
- Update to latest Mathematics package version

## [3.0.2] - 2019-07-13
### Changed
- Mark package to support Unity 2019.3.0a10 onwards.

## [3.0.1] - 2019-07-12
### Changed
- Fix path length due to validation failure.

## [3.0.0] - 2019-06-17
### Changed
- Remove preview tag.
- Remove experimental namespace

## [2.2.0-preview.3] - 2019-06-06
### Added
- BoneGizmos can be toggled in the SceneView Gizmos dropdown
- Scrollbar does not show for toolbar when required to scroll
- Change Sprite Library implementation.
- APIs redesigned. This will cause compilation errors if previous APIs are used
- Data serialization change. Previous Asset and Component data will not be compatible

## [2.2.0-preview.2] - 2019-05-10
### Added
- BoneGizmos will only show in selected hierarchies.
- Associate nearest Bone to Sprite intead of root Bone when no Bones are overlapping the Sprite
- Fixed Sprite not showing after it is being culled due to bone animation
- Add icons for Sprite Library Asset, Sprite Library Component and Sprite Resolver Component
- Fixed Sprite Library Asset Inspector Property Field text clipping
- SpriteResolver will assign Sprite to SpriteRenderer even it resolves to a missing Sprite
- Add visual feedback in SpriteResolver Inspector for missing Sprite

## [2.2.0-preview.1] - 2019-05-09
### Added
- Upgrade for 2019.2
- Copy and Paste rework
- Visibility Window remains open when switching between tools
- Reparent Bone tool removed and functionality moved into Bone Visibility Panel
- Added Sprite Library feature
- Add Layer Grouping support in Sprite Visibility Panel

## [2.1.0-preview.4] - 2019-04-29
### Added
- Fix skinning not in sync with the rendering.

## [2.1.0-preview.3] - 2019-04-24
### Added
- Set Burst compilation off for internal build

## [2.1.0-preview.2] - 2019-02-25
### Added
- Fix enable skinning on add SpriteSkin component
- Upgrade dependency package version for Unity 2019.1 support
- Fix case 1118093: SpriteSkin.onDrawGizmos() increases memory usage.

## [2.1.0-preview.1] - 2019-01-25
### Added
- Update package to work with 2019.1
- Improve animation runtime performance
- Fix bone reparenting sibling order
- Fix Sprite Visibility Tool in disabled state in certain cases
- Update documents

## [2.0.0-preview.1] - 2018-11-20
### Added
- Overhauled 2D Animation workflow.
  - Refer to updated documentation for workflow changes.
- Single Sprite Editor Window module for 2D Sprite Rigging workflow
  - Unified Bone, Geometry and Weight tools in a single window
- Supports Multiple Sprite Single Character rigging workflow through 2D PSD Importer Package.
- SpriteSkin now uses user define bounds for renderer culling

## [1.0.16-preview.2] - 2018-11-14
### Added
- Fix 2 Issues:
  1. Prefabs with SpriteSkin loses references to bone hierarchy when Library folder is rebuilt/different.
  2. The scene viewport shows the character without any bones applied, needing an re-import.

## [1.0.16-preview.1] - 2018-07-18
### Added
- Fix error log about VertexAttribute

## [1.0.16-preview] - 2018-06-20
### Added
- Fix Documentation warnings
- Fix error log complaining about DidReloadScripts signature.
- Fix issues with generate outline

## [1.0.15-preview] - 2018-04-12
### Added
- New Version suffix (preview)
- Improved Scene View gizmos for better manipulation of bone rotation and position
- Added notification when Sprites are imported with incorrect weights
- Fixed bug where textures with max texture size could not generate geometry
