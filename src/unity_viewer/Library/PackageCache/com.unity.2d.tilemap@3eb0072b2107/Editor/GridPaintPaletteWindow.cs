using System;
using System.Collections.Generic;
using UnityEditor.EditorTools;
using UnityEditor.Overlays;
using UnityEditor.ShortcutManagement;
using UnityEngine;
using UnityEngine.UIElements;

using Event = UnityEngine.Event;

namespace UnityEditor.Tilemaps
{
    /// <summary>
    /// EditorWindow containing the Tile Palette
    /// </summary>
    public class GridPaintPaletteWindow : EditorWindow, ISupportsOverlays
    {
        private static class Styles
        {
            public static readonly GUIContent selectPaintTarget = EditorGUIUtility.TrTextContent("Select Paint Target");
            public static readonly GUIContent selectPalettePrefab = EditorGUIUtility.TrTextContent("Select Palette Prefab");
            public static readonly GUIContent selectTileAsset = EditorGUIUtility.TrTextContent("Select Tile Asset");
            public static readonly GUIContent unlockPaletteEditing = EditorGUIUtility.TrTextContent("Unlock Palette Editing");
            public static readonly GUIContent lockPaletteEditing = EditorGUIUtility.TrTextContent("Lock Palette Editing");
            public static readonly GUIContent verticalBrushSplit = EditorGUIUtility.TrTextContent("Vertical Split for Brush Inspector");
            public static readonly GUIContent horizontalBrushSplit = EditorGUIUtility.TrTextContent("Horizontal Split for Brush Inspector");
            public static readonly GUIContent openTilePalettePreferences = EditorGUIUtility.TrTextContent("Open Tile Palette Preferences");
            public static readonly GUIContent openAsFloatingWindow = EditorGUIUtility.TrTextContent("Open Window as/Floating");
            public static readonly GUIContent openAsDockableWindow = EditorGUIUtility.TrTextContent("Open Window as/Dockable");

            public static readonly GUIContent tilePalette = EditorGUIUtility.TrTextContent("Tile Palette");

            public static readonly GUIContent mouseGridPositionAtZ = EditorGUIUtility.TrTextContent("Mouse Grid Position At Z", "Shows the Mouse Grid Position marquee at the Brush's Z Position.");
        }

        private static class UIStyles
        {
            public static readonly string styleSheetPath = "Packages/com.unity.2d.tilemap/Editor/UI/GridPaintPaletteWindow.uss";
            public static readonly string darkStyleSheetPath = "Packages/com.unity.2d.tilemap/Editor/UI/GridPaintPaletteWindowDark.uss";
            public static readonly string lightStyleSheetPath = "Packages/com.unity.2d.tilemap/Editor/UI/GridPaintPaletteWindowLight.uss";
            public static readonly string ussClassName = "unity-grid-paint-palette-window";
        }

        private static readonly string k_TilePaletteVerticalBrushSplitPref = "TilePaletteVerticalBrushSplit";
        internal static bool tilePaletteVerticalBrushSplit
        {
            get
            {
                return EditorPrefs.GetBool(k_TilePaletteVerticalBrushSplitPref, true);
            }
            set
            {
                EditorPrefs.SetBool(k_TilePaletteVerticalBrushSplitPref, value);
            }
        }

        private const float k_ActiveTargetLabelWidth = 90f;
        private const float k_ActiveTargetDropdownWidth = 130f;
        private const float k_ActiveTargetWarningSize = 20f;
        private const float k_SceneViewToggleSize = 68f;
        private const float k_MinClipboardHeight = 200f;
        private static readonly Vector2 k_MinWindowSize = new Vector2(k_ActiveTargetLabelWidth + k_ActiveTargetDropdownWidth + k_ActiveTargetWarningSize + k_SceneViewToggleSize, k_MinClipboardHeight);

        internal static class ShortcutIds
        {
            public const string k_Select = "Grid Painting/Select";
            public const string k_Move = "Grid Painting/Move";
            public const string k_Brush = "Grid Painting/Brush";
            public const string k_Rectangle = "Grid Painting/Rectangle";
            public const string k_Picker = "Grid Painting/Picker";
            public const string k_Erase = "Grid Painting/Erase";
            public const string k_Fill = "Grid Painting/Fill";
            public const string k_RotateClockwise = "Grid Painting/Rotate Clockwise";
            public const string k_RotateAntiClockwise = "Grid Painting/Rotate Anti-Clockwise";
            public const string k_FlipX = "Grid Painting/Flip X";
            public const string k_FlipY = "Grid Painting/Flip Y";
            public const string k_IncreaseZ = "Grid Painting/Increase Z";
            public const string k_DecreaseZ = "Grid Painting/Decrease Z";
            public const string k_SwitchToNextBrush = "Grid Painting/Switch To Next Brush";
            public const string k_SwitchToPreviousBrush = "Grid Painting/Switch To Previous Brush";
            public const string k_ToggleSceneViewPalette = "Grid Painting/Toggle SceneView Palette";
            public const string k_ToggleSceneViewBrushPick = "Grid Painting/Toggle SceneView BrushPick";
        }

        [FormerlyPrefKeyAs(ShortcutIds.k_Select, "s")]
        [Shortcut(ShortcutIds.k_Select, typeof(TilemapEditorTool.ShortcutContext), KeyCode.S)]
        static void GridSelectKey()
        {
            TilemapEditorTool.ToggleActiveEditorTool(typeof(SelectTool));
        }

        [FormerlyPrefKeyAs(ShortcutIds.k_Move, "m")]
        [Shortcut(ShortcutIds.k_Move, typeof(TilemapEditorTool.ShortcutContext), KeyCode.M)]
        static void GridMoveKey()
        {
            TilemapEditorTool.ToggleActiveEditorTool(typeof(MoveTool));
        }

        [FormerlyPrefKeyAs(ShortcutIds.k_Brush, "b")]
        [Shortcut(ShortcutIds.k_Brush, typeof(TilemapEditorTool.ShortcutContext), KeyCode.B)]
        static void GridBrushKey()
        {
            TilemapEditorTool.ToggleActiveEditorTool(typeof(PaintTool));
        }

        [FormerlyPrefKeyAs(ShortcutIds.k_Rectangle, "u")]
        [Shortcut(ShortcutIds.k_Rectangle, typeof(TilemapEditorTool.ShortcutContext), KeyCode.U)]
        static void GridRectangleKey()
        {
            TilemapEditorTool.ToggleActiveEditorTool(typeof(BoxTool));
        }

        [FormerlyPrefKeyAs(ShortcutIds.k_Picker, "i")]
        [Shortcut(ShortcutIds.k_Picker, typeof(TilemapEditorTool.ShortcutContext), KeyCode.I)]
        static void GridPickerKey()
        {
            TilemapEditorTool.ToggleActiveEditorTool(typeof(PickingTool));
        }

        [FormerlyPrefKeyAs(ShortcutIds.k_Erase, "d")]
        [Shortcut(ShortcutIds.k_Erase, typeof(TilemapEditorTool.ShortcutContext), KeyCode.D)]
        static void GridEraseKey()
        {
            TilemapEditorTool.ToggleActiveEditorTool(typeof(EraseTool));
        }

        [FormerlyPrefKeyAs(ShortcutIds.k_Fill, "g")]
        [Shortcut(ShortcutIds.k_Fill, typeof(TilemapEditorTool.ShortcutContext), KeyCode.G)]
        static void GridFillKey()
        {
            TilemapEditorTool.ToggleActiveEditorTool(typeof(FillTool));
        }

        static void RotateBrush(GridBrushBase.RotationDirection direction)
        {
            GridPaintingState.gridBrush.Rotate(direction, GridPaintingState.activeGrid.cellLayout);
            GridPaintingState.activeGrid.Repaint();
        }

        [FormerlyPrefKeyAs(ShortcutIds.k_RotateClockwise, "]")]
        [Shortcut(ShortcutIds.k_RotateClockwise, typeof(TilemapEditorTool.ShortcutContext), KeyCode.RightBracket)]
        static void RotateBrushClockwise()
        {
            if (GridPaintingState.gridBrush != null && GridPaintingState.activeGrid != null)
                RotateBrush(GridBrushBase.RotationDirection.Clockwise);
        }

        [FormerlyPrefKeyAs(ShortcutIds.k_RotateAntiClockwise, "[")]
        [Shortcut(ShortcutIds.k_RotateAntiClockwise, typeof(TilemapEditorTool.ShortcutContext), KeyCode.LeftBracket)]
        static void RotateBrushAntiClockwise()
        {
            if (GridPaintingState.gridBrush != null && GridPaintingState.activeGrid != null)
                RotateBrush(GridBrushBase.RotationDirection.CounterClockwise);
        }

        static void FlipBrush(GridBrushBase.FlipAxis axis)
        {
            GridPaintingState.gridBrush.Flip(axis, GridPaintingState.activeGrid.cellLayout);
            GridPaintingState.activeGrid.Repaint();
        }

        [FormerlyPrefKeyAs(ShortcutIds.k_FlipX, "#[")]
        [Shortcut(ShortcutIds.k_FlipX, typeof(TilemapEditorTool.ShortcutContext), KeyCode.LeftBracket, ShortcutModifiers.Shift)]
        static void FlipBrushX()
        {
            if (GridPaintingState.gridBrush != null && GridPaintingState.activeGrid != null)
                FlipBrush(GridBrushBase.FlipAxis.X);
        }

        [FormerlyPrefKeyAs(ShortcutIds.k_FlipY, "#]")]
        [Shortcut(ShortcutIds.k_FlipY, typeof(TilemapEditorTool.ShortcutContext), KeyCode.RightBracket, ShortcutModifiers.Shift)]
        static void FlipBrushY()
        {
            if (GridPaintingState.gridBrush != null && GridPaintingState.activeGrid != null)
                FlipBrush(GridBrushBase.FlipAxis.Y);
        }

        static void ChangeBrushZ(int change)
        {
            GridPaintingState.gridBrush.ChangeZPosition(change);
            GridPaintingState.activeGrid.ChangeZPosition(change);
            GridPaintingState.activeGrid.Repaint();

            foreach (var window in instances)
            {
                window.Repaint();
            }
        }

        [Shortcut(ShortcutIds.k_IncreaseZ, typeof(TilemapEditorTool.ShortcutContext), KeyCode.Minus)]
        static void IncreaseBrushZ()
        {
            if (GridPaintingState.gridBrush != null
                && GridPaintingState.activeGrid != null
                && GridPaintingState.activeBrushEditor != null
                && GridPaintingState.activeBrushEditor.canChangeZPosition)
                ChangeBrushZ(1);
        }

        [Shortcut(ShortcutIds.k_DecreaseZ, typeof(TilemapEditorTool.ShortcutContext), KeyCode.Equals)]
        static void DecreaseBrushZ()
        {
            if (GridPaintingState.gridBrush != null
                && GridPaintingState.activeGrid != null
                && GridPaintingState.activeBrushEditor != null
                && GridPaintingState.activeBrushEditor.canChangeZPosition)
                ChangeBrushZ(-1);
        }

        [Shortcut(ShortcutIds.k_SwitchToNextBrush, typeof(TilemapEditorTool.ShortcutContext), KeyCode.B, ShortcutModifiers.Shift)]
        static void SwitchToNextBrush()
        {
            SwitchBrush(1);
        }

        [Shortcut(ShortcutIds.k_SwitchToPreviousBrush, typeof(TilemapEditorTool.ShortcutContext), KeyCode.B, ShortcutModifiers.Shift | ShortcutModifiers.Alt)]
        static void SwitchToPreviousBrush()
        {
            SwitchBrush(-1);
        }

        static void SwitchBrush(int change)
        {
            var count = GridPaintingState.brushes.Count;
            var index = GridPaintingState.brushes.IndexOf(GridPaintingState.gridBrush);
            var newIndex = (index + change + count) % count;
            if (index != newIndex)
                GridPaintingState.gridBrush = GridPaintingState.brushes[newIndex];
        }

        [Shortcut(ShortcutIds.k_ToggleSceneViewPalette, typeof(TilemapEditorTool.ShortcutContext), KeyCode.Semicolon)]
        static void ToggleSceneViewPalette()
        {
            var sceneView = SceneView.lastActiveSceneView;
            if (sceneView == null)
                return;

            if (!sceneView.TryGetOverlay(TilePaletteClipboardOverlay.k_OverlayId, out Overlay overlay))
                return;

            var tilePaletteClipboardOverlay = overlay as TilePaletteClipboardOverlay;
            if (tilePaletteClipboardOverlay == null)
                return;

            tilePaletteClipboardOverlay.TogglePopup(GridPaintingState.lastSceneViewMousePosition);
        }

        [Shortcut(ShortcutIds.k_ToggleSceneViewBrushPick, typeof(TilemapEditorTool.ShortcutContext), KeyCode.Quote)]
        static void ToggleSceneViewBrushPick()
        {
            var sceneView = SceneView.lastActiveSceneView;
            if (sceneView == null)
                return;

            if (!sceneView.TryGetOverlay(TilePaletteBrushPickOverlay.k_OverlayId, out Overlay overlay))
                return;

            var tilePaletteBrushPickOverlay = overlay as TilePaletteBrushPickOverlay;
            if (tilePaletteBrushPickOverlay == null)
                return;

            tilePaletteBrushPickOverlay.TogglePopup(GridPaintingState.lastSceneViewMousePosition);
        }

        internal static void PreferencesGUI()
        {
            using (new SettingsWindow.GUIScope())
            {
                EditorGUI.BeginChangeCheck();
                var val = (TilePaletteActiveTargetsProperties.PrefabEditModeSettings)EditorGUILayout.EnumPopup(TilePaletteActiveTargetsProperties.targetEditModeDialogLabel
                    , (TilePaletteActiveTargetsProperties.PrefabEditModeSettings)EditorPrefs.GetInt(TilePaletteActiveTargetsProperties.targetEditModeEditorPref
                        , 0));
                if (EditorGUI.EndChangeCheck())
                {
                    EditorPrefs.SetInt(TilePaletteActiveTargetsProperties.targetEditModeEditorPref, (int)val);
                }
                EditorGUI.BeginChangeCheck();
                var val2 = EditorGUILayout.Toggle(Styles.mouseGridPositionAtZ, GridPaintingState.gridBrushMousePositionAtZ);
                if (EditorGUI.EndChangeCheck())
                {
                    GridPaintingState.gridBrushMousePositionAtZ = val2;
                }
            }
        }

        private static List<GridPaintPaletteWindow> s_Instances;

        private static List<GridPaintPaletteWindow> instances
        {
            get
            {
                if (s_Instances == null)
                    s_Instances = new List<GridPaintPaletteWindow>();
                return s_Instances;
            }
        }

        /// <summary>
        /// Whether the GridPaintPaletteWindow is active in the Editor
        /// </summary>
        public static bool isActive
        {
            get
            {
                return s_Instances != null && s_Instances.Count > 0;
            }
        }

        internal GameObject palette
        {
            get => GridPaintingState.palette;
            set => GridPaintingState.palette = value;
        }

        internal GameObject paletteInstance => clipboardView.paletteInstance;

        internal GridPaintPaletteClipboard clipboardView
        {
            get => m_ClipboardSplitView.paletteElement.clipboardView;
        }

        private Vector2 m_BrushScroll;
        private bool m_IsUtilityWindow;

        private VisualElement m_ToolbarVisualElement;
        private VisualElement m_ActiveTargetsVisualElement;
        private GridPaintPaletteWindowSplitView m_ClipboardSplitView;

        private void CreateGUI()
        {
            var styleSheet = AssetDatabase.LoadAssetAtPath<StyleSheet>(UIStyles.styleSheetPath);
            var skinStyleSheet = EditorGUIUtility.isProSkin
                ? AssetDatabase.LoadAssetAtPath<StyleSheet>(UIStyles.darkStyleSheetPath)
                : AssetDatabase.LoadAssetAtPath<StyleSheet>(UIStyles.lightStyleSheetPath);
            if (styleSheet == null || skinStyleSheet == null)
                return;

            m_ToolbarVisualElement = new GridPaintingToolbar(this);
            m_ActiveTargetsVisualElement = new GridPaintPaletteWindowActiveTargets()
            {
                name = "activeTargetsDropdown",
            };
            m_ClipboardSplitView = new GridPaintPaletteWindowSplitView(tilePaletteVerticalBrushSplit);

            var root = rootVisualElement;
            root.Add(m_ToolbarVisualElement);
            root.Add(m_ActiveTargetsVisualElement);
            root.Add(m_ClipboardSplitView);

            root.styleSheetList.Add(styleSheet);
            root.styleSheetList.Add(skinStyleSheet);
            root.AddToClassList(UIStyles.ussClassName);
            root.style.minHeight = k_MinClipboardHeight;

            root.AddManipulator(new TilePaletteContextMenuHandler(DoContextMenu));
            m_ToolbarVisualElement.AddManipulator(new TilePaletteContextMenuHandler(DoContextMenu));
            m_ActiveTargetsVisualElement.AddManipulator(new TilePaletteContextMenuHandler(DoContextMenu));

            m_ClipboardSplitView.AddManipulator(new TilePaletteDragHandler(DragUpdatedForConvertGridPrefabToPalette, DragPerformedForConvertGridPrefabToPalette));

            UpdateWindowMinSize();
        }

        private void OnGUI()
        {
            // Keep repainting until all previews are loaded
            if (AssetPreview.IsLoadingAssetPreviews())
                Repaint();

            // Release keyboard focus on click to empty space
            if (Event.current.type == EventType.MouseDown)
                GUIUtility.keyboardControl = 0;
        }

        private void DoContextMenu()
        {
            GenericMenu pm = new GenericMenu();
            if (GridPaintingState.scenePaintTarget != null)
                pm.AddItem(Styles.selectPaintTarget, false, SelectPaintTarget);
            else
                pm.AddDisabledItem(Styles.selectPaintTarget);

            if (palette != null)
                pm.AddItem(Styles.selectPalettePrefab, false, SelectPaletteAsset);
            else
                pm.AddDisabledItem(Styles.selectPalettePrefab);

            if (clipboardView.activeTile != null)
                pm.AddItem(Styles.selectTileAsset, false, SelectTileAsset);
            else
                pm.AddDisabledItem(Styles.selectTileAsset);

            pm.AddSeparator("");

            if (!GridPaintingState.isPaletteEditable)
                pm.AddDisabledItem(Styles.unlockPaletteEditing);
            else if (clipboardView.unlocked)
                pm.AddItem(Styles.lockPaletteEditing, false, FlipLocked);
            else
                pm.AddItem(Styles.unlockPaletteEditing, false, FlipLocked);

            if (tilePaletteVerticalBrushSplit)
                pm.AddItem(Styles.horizontalBrushSplit, false, FlipShowToolbarInSceneView);
            else
                pm.AddItem(Styles.verticalBrushSplit, false, FlipShowToolbarInSceneView);

            pm.AddItem(Styles.openTilePalettePreferences, false, OpenTilePalettePreferences);

            pm.AddItem(Styles.openAsDockableWindow, !m_IsUtilityWindow, () => OpenWindow(false));
            pm.AddItem(Styles.openAsFloatingWindow, m_IsUtilityWindow, () => OpenWindow(true));

            pm.ShowAsContext();
        }

        private void OpenWindow(bool utility)
        {
            Close();
            GridPaintPaletteWindow w = GetWindow<GridPaintPaletteWindow>(utility, Styles.tilePalette.text, true);
            w.m_IsUtilityWindow = utility;
        }

        private void OpenTilePalettePreferences()
        {
            var settingsWindow = SettingsWindow.Show(SettingsScope.User);
            settingsWindow.FilterProviders(TilePaletteActiveTargetsProperties.tilePalettePreferencesLookup);
        }

        private void FlipLocked()
        {
            m_ClipboardSplitView.paletteElement.clipboardUnlocked = !m_ClipboardSplitView.paletteElement.clipboardUnlocked;
        }

        private void FlipShowToolbarInSceneView()
        {
            var state = !m_ClipboardSplitView.isVerticalOrientation;
            tilePaletteVerticalBrushSplit = state;
            m_ClipboardSplitView.isVerticalOrientation = state;

            SceneView.RepaintAll();
        }

        private void SelectPaintTarget()
        {
            Selection.activeObject = GridPaintingState.scenePaintTarget;
        }

        private void SelectPaletteAsset()
        {
            Selection.activeObject = palette;
        }

        private void SelectTileAsset()
        {
            Selection.activeObject = clipboardView.activeTile;
        }

        internal void OnEnable()
        {
            instances.Add(this);

            GridSelection.gridSelectionChanged += OnGridSelectionChanged;
            EditorApplication.projectWasLoaded += OnProjectLoaded;
            ToolManager.activeToolChanged += ActiveToolChanged;
            TilemapEditorToolPreferences.tilemapEditorToolsChanged += UpdateWindowMinSize;
            GridPaintingState.brushToolsChanged += UpdateWindowMinSize;

            wantsMouseMove = true;
            wantsMouseEnterLeaveWindow = true;

            GridPaintingState.RegisterPainterInterest(this);
        }

        private void OnProjectLoaded()
        {
            GridPaintingState.RegisterShortcutContext();
        }

        private void OnGridSelectionChanged()
        {
            Repaint();
        }

        private void UpdateWindowMinSize()
        {
            var minWindowSize = k_MinWindowSize;

            var tools = TilemapEditorTool.tilemapEditorTools;
            var width = tools != null ? tools.Length * 32f + 4f : 0f;

            // Calculate min width from tools and fixed active targets
            minWindowSize.x = Math.Max(minWindowSize.x, width);

            minSize = minWindowSize;
        }

        internal void OnDisable()
        {
            GridPaintingState.UnregisterPainterInterest(this);

            TilemapEditorToolPreferences.tilemapEditorToolsChanged -= UpdateWindowMinSize;
            GridPaintingState.brushToolsChanged -= UpdateWindowMinSize;
            ToolManager.activeToolChanged -= ActiveToolChanged;
            GridSelection.gridSelectionChanged -= OnGridSelectionChanged;
            EditorApplication.projectWasLoaded -= OnProjectLoaded;

            instances.Remove(this);
        }

        private void ActiveToolChanged()
        {
            Repaint();
        }

        private bool ValidateDragAndDrop()
        {
            if (DragAndDrop.objectReferences.Length != 1)
                return false;

            var draggedObject = DragAndDrop.objectReferences[0];
            if (!PrefabUtility.IsPartOfRegularPrefab(draggedObject))
                return false;

            return true;
        }

        private void DragUpdatedForConvertGridPrefabToPalette()
        {
            if (!ValidateDragAndDrop())
                return;

            DragAndDrop.visualMode = DragAndDropVisualMode.Generic;
        }

        private void DragPerformedForConvertGridPrefabToPalette()
        {
            if (!ValidateDragAndDrop())
                return;

            var draggedObject = DragAndDrop.objectReferences[0];
            var path = AssetDatabase.GetAssetPath(draggedObject);
            var assets = AssetDatabase.LoadAllAssetsAtPath(path);
            bool hasNewPaletteAsset = false;
            Grid gridPrefab = null;
            foreach (var asset in assets)
            {
                var gridPalette = asset as GridPalette;
                hasNewPaletteAsset |= gridPalette != null;
                GameObject go = asset as GameObject;
                if (go != null)
                {
                    var grid = go.GetComponent<Grid>();
                    if (grid != null)
                        gridPrefab = grid;
                }
            }
            if (!hasNewPaletteAsset && gridPrefab != null)
            {
                var cellLayout = gridPrefab.cellLayout;
                var cellSizing = (cellLayout == GridLayout.CellLayout.Rectangle
                    || cellLayout == GridLayout.CellLayout.Hexagon)
                    ? GridPalette.CellSizing.Automatic
                    : GridPalette.CellSizing.Manual;
                var newPalette = GridPaletteUtility.CreateGridPalette(cellSizing);
                AssetDatabase.AddObjectToAsset(newPalette, path);
                AssetDatabase.ForceReserializeAssets(new[] {path});
                AssetDatabase.SaveAssets();
                Event.current.Use();
                GUIUtility.ExitGUI();
            }
        }

        internal void ResetZPosition()
        {
            GridPaintingState.gridBrush.ResetZPosition();
            GridPaintingState.lastActiveGrid.ResetZPosition();
        }

        [MenuItem("Window/2D/Tile Palette", false, 2)]
        internal static void OpenTilemapPalette()
        {
            GridPaintPaletteWindow w = GetWindow<GridPaintPaletteWindow>();
            w.titleContent = Styles.tilePalette;
            w.m_IsUtilityWindow = false;
        }
    }
}
