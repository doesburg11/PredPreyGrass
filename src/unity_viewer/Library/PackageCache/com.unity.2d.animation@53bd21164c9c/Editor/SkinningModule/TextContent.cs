using UnityEngine;

namespace UnityEditor.U2D.Animation
{
    internal static class TextContent
    {
        // Undo
        public static string setMode = "Set Mode";
        public static string setTool = "Set Tool";
        public static string pasteData = "Paste Data";
        public static string generateGeometry = "Generate Geometry";
        public static string generateWeights = "Generate Weights";
        public static string normalizeWeights = "Normalize Weights";
        public static string clearWeights = "Clear Weights";
        public static string restorePose = "Restore Pose";
        public static string pivotPose = "Pivot Pose";
        public static string selection = "Selection";
        public static string clearSelection = "Clear Selection";
        public static string editWeights = "Edit Weights";
        public static string boneName = "Bone Name";
        public static string boneDepth = "Bone Depth";
        public static string boneColor = "Bone Color";
        public static string rotateBone = "Rotate Bone";
        public static string moveBone = "Move Bone";
        public static string colorBoneChanged = "Bone Color";
        public static string freeMoveBone = "Free Move Bone";
        public static string moveJoint = "Move Joint";
        public static string moveEndPoint = "Move End Point";
        public static string boneLength = "Bone Length";
        public static string createBone = "Create Bone";
        public static string splitBone = "Split Bone";
        public static string removeBone = "Remove Bone";
        public static string moveVertices = "Move Vertices";
        public static string createVertex = "Create Vertex";
        public static string createEdge = "Create Edge";
        public static string splitEdge = "Split Edge";
        public static string removeEdge = "Remove Edge";
        public static string removeVertices = "Remove Vertices";
        public static string selectionChange = "Selection Change";
        public static string boneVisibility = "Bone Visibility";
        public static string setParentBone = "Set Parent Bone";
        public static string visibilityChange = "VisibilityChange";
        public static string boneSelection = "Bone Selection";
        public static string expandBones = "Expand Bones";
        public static string meshVisibility = "Mesh Visibility";
        public static string meshOpacity = "Mesh Opacity";
        public static string opacityChange = "Opacity Change";
        public static string spriteVisibility = "SpriteVisibility";
        public static string visibilityTab = "Visibility Tab";
        public static string addBoneInfluence = "Add Bone Influence";
        public static string removeBoneInfluence = "Remove Bone Influence";
        public static string reorderBoneInfluence = "Reorder Bone Influence";
        public static string addSpriteInfluence = "Add Sprite Influence";
        public static string removeSpriteInfluence = "Remove Sprite Influence";
        public static string spriteSelection = "Sprite selection";
        public static string pivotChanged = "Pivot changed";

        // Tooltips
        public static string visibilityIconTooltip = L10n.Tr("Visibility tool");
        public static string characterIconTooltip = L10n.Tr("Restore bind pose");
        public static string spriteSheetIconTooltip = L10n.Tr("Switch between Sprite sheet and Character mode");
        public static string copyTooltip = L10n.Tr("Copy");
        public static string pasteTooltip = L10n.Tr("Paste");
        public static string onTooltip = L10n.Tr("On");
        public static string offTooltip = L10n.Tr("Off");
        public static string addBoneInfluenceTooltip = L10n.Tr("Add selected bone to influence Sprite");
        public static string removeBoneInfluenceTooltip = L10n.Tr("Remove selected bone from influencing Sprite");
        public static string addSpriteInfluenceTooltip = L10n.Tr("Add selected Sprite to be influenced by bone");
        public static string removeSpriteInfluenceTooltip = L10n.Tr("Remove selected Sprite from being influenced by bone");

        // Horizontal tool bar button txt
        public static string visibilityIconText = L10n.Tr("Visibility");
        public static string spriteSheetIconText = L10n.Tr("Sprite Sheet");
        public static string copyText = L10n.Tr("Copy");
        public static string pasteText = L10n.Tr("Paste");

        // Settings
        public static string selectedOutlineColor = L10n.Tr("Selected Outline Color");
        public static string spriteOutlineSize = L10n.Tr("Sprite Outline Size");
        public static string boneOutlineSize = L10n.Tr("Bone Outline Size");
        public static string showSpriteMeshOverwriteWarning = L10n.Tr("Mesh Overwrite Warning");
        public static string showSpriteMeshOverwriteWarningTip = L10n.Tr("Log warning message when Skinning Editor is overwriting custom outline mesh.");

        // Sprite Library
        public static string categoryList = L10n.Tr("Category List");
        public static string category = L10n.Tr("Category");
        public static string label = L10n.Tr("Label");
        public static string spriteLibraryCircularDependency = L10n.Tr("The Sprite Library can't be added because self-referencing is not allowed. Try another Sprite Library instead.");
        public static string spriteLibraryNoAssetSelected = L10n.Tr("Select a Sprite Library Asset in your Project window.\n\nNew Sprite Library Assets can be created from the asset creation menu:\nAsset > Create > 2D > Sprite Library Asset");
        public static string spriteCategoryColumnEmpty = L10n.Tr("To start creating categories drag and drop Sprites or Sprite Texture assets or select the '+' button.");
        public static string spriteLabelColumnEmpty = L10n.Tr("To start creating labels drag and drop Sprites or Sprite Texture or select the '+' button.");
        public static string spriteCategoryMultiSelect = L10n.Tr("Multiple Categories selected.");
        public static string spriteCategoryNoSelection = L10n.Tr("No Categories selected.");
        public static string spriteLibraryRevertMessage = L10n.Tr("There are some unsaved changes, are you sure you want to revert them?");
        public static string spriteLibraryMainLibraryTooltip = L10n.Tr("This field is optional. By linking a Main Library, this Sprite Library becomes a Variant of the Main Library allowing it to reference all the Main Library’s Categories.");
        public static string spriteLibraryCategoriesTooltip = L10n.Tr("A container to organize the Labels. The Category must be unique from other Categories in the same Sprite Library or Sprite Library hierarchy.");
        public static string spriteLibraryLabelsTooltip = L10n.Tr("Label contains a Sprite reference. Name has to be unique.");
        public static string spriteLibraryAddCategoryTooltip = L10n.Tr("Click here to add a new Category.");
        public static string spriteLibraryAddCategoryTooltipNotAvailable = L10n.Tr("Cannot create new Category with search filter selected.");
        public static string spriteLibraryAddLabelTooltip = L10n.Tr("Click here to add a new Label.");
        public static string spriteLibraryAddLabelTooltipNotAvailable = L10n.Tr("Cannot create new Label with search filter selected.");
        public static string spriteLibraryLocalCategoryTooltip = L10n.Tr("Local Categories exits only in this Sprite Library Asset.");
        public static string spriteLibraryInheritedCategoryTooltip = L10n.Tr("Inherited Categories come from the Main Library Asset. Their Labels can be overriden in the Labels tab.");
        public static string spriteLibraryCreateCategory = L10n.Tr("Create new Category");
        public static string spriteLibraryRenameCategory = L10n.Tr("Rename Category");
        public static string spriteLibraryDeleteCategories = L10n.Tr("Delete Selected Categories");
        public static string spriteLibraryCreateLabel = L10n.Tr("Create new Label");
        public static string spriteLibraryRenameLabel = L10n.Tr("Rename Label");
        public static string spriteLibraryDeleteLabels = L10n.Tr("Delete Selected Labels");
        public static string spriteLibraryShowLabel = L10n.Tr("Show Label Location");
        public static string spriteLibraryRevertLabels = L10n.Tr("Revert Selected Overrides");
        public static string spriteLibraryRevertAllLabels = L10n.Tr("Revert All Overrides");
        public static string spriteLibrarySetMainLibrary = L10n.Tr("Set Main Library");
        public static string spriteLibrarySelectCategories = L10n.Tr("Select Categories");
        public static string spriteLibrarySelectLabels = L10n.Tr("Select Labels");
        public static string spriteLibraryReorderCategories = L10n.Tr("Reorder Categories");
        public static string spriteLibraryReorderLabels = L10n.Tr("Reorder Labels");
        public static string spriteLibrarySetLabelSprite = L10n.Tr("Set Label's Sprite");
        public static string spriteLibraryAddDataToCategories = L10n.Tr("Add data to Categories");
        public static string spriteLibraryAddDataToLabels = L10n.Tr("Add data to Labels");
        public static string spriteLibraryCreateNewAsset = L10n.Tr("Create new Sprite Library Asset");
        public static string spriteLibraryCreateTitle = L10n.Tr("New Sprite Library Asset");
        public static string spriteLibraryCreateMessage = L10n.Tr("Create a new Sprite Library Asset");

        // Sprite Resolver
        public static readonly string emptyCategory = L10n.Tr("Category is Empty");
        public static readonly string noCategory = L10n.Tr("No Category");
        public static readonly string spriteSwapSelectSpriteResolver = L10n.Tr("Select any Game Object(s) with the Sprite Resolver component in the Scene to start Sprite Swapping.");
        public static readonly string spriteSwapFilterDescription = L10n.Tr("Filters out single-label Sprite Resolvers");
        public static readonly string spriteSwapLockDescription = L10n.Tr("Locks the current selection");
        public static readonly string spriteSwapResetThumbnailSize = L10n.Tr("Resets the Label thumbnail to its default size");
        public static readonly string spriteSwapThumbnailSlider = L10n.Tr("Sets the Label thumbnail size");
        public static readonly string spriteSwapFilteredContent = L10n.Tr("Filtered Content");

        // Other
        public static string savePopupTitle = L10n.Tr("Unsaved changes");
        public static string savePopupMessage = L10n.Tr("There are some unsaved changes, would you like to save them?");
        public static string savePopupOptionYes = L10n.Tr("Yes");
        public static string savePopupOptionNo = L10n.Tr("No");
        public static string generatingOutline = L10n.Tr("Generating Outline");
        public static string triangulatingGeometry = L10n.Tr("Triangulating Geometry");
        public static string subdividingGeometry = L10n.Tr("Subdividing Geometry");
        public static string generatingWeights = L10n.Tr("Generating Weights");
        public static string restorePoseLocalized = L10n.Tr("Restore Pose");
        public static string vertexWeight = L10n.Tr("Vertex Weight");
        public static string vertexWeightToolTip = L10n.Tr("Adjust bone weights for selected vertex");
        public static string bone = L10n.Tr("Bone");
        public static string depth = L10n.Tr("Depth");
        public static string color = L10n.Tr("Color");
        public static string sprite = L10n.Tr("Sprite");
        public static string name = L10n.Tr("Name");
        public static string none = L10n.Tr("None");
        public static string size = L10n.Tr("Size");
        public static string noSpriteSelected = L10n.Tr("No sprite selected");
        public static string weightSlider = L10n.Tr("Weight Slider");
        public static string weightBrush = L10n.Tr("Weight Brush");
        public static string generateAll = L10n.Tr("Generate All");
        public static string generate = L10n.Tr("Generate");
        public static string mode = L10n.Tr("Mode");
        public static string modeTooltip = L10n.Tr("Different operation mode for weight adjustment");
        public static string boneToolTip = L10n.Tr("The bone that is affecting");
        public static string pivot = L10n.Tr("Pivot");

        public static string noBoneSelected = L10n.Tr("No bone selected");
        public static string boneInfluences = L10n.Tr("Bone Influences");
        public static string influencedSprites = L10n.Tr("Sprite Influences");

        // Error messages
        public static string copyIncorrectNumberOfSprites = L10n.Tr("Cannot paste Sprites ({0}) from the source, because the target has a different number of Sprites ({1}).");
        public static string spriteMeshOverwriteWarning = L10n.Tr("{0} has custom outline defined. The Sprite's mesh will be overwritten by the mesh defined in Skinning Editor.");
        public static string boneWeightsNotSumZeroWarning = L10n.Tr("Sprite {0} contains bone weights which sum zero or are not normalized. To avoid visual artifacts please consider fixing them.");
    }
}
