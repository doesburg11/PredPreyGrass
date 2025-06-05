using System;
using System.Collections.Generic;
using UnityEditor.EditorTools;
using UnityEngine;
using UnityEngine.Tilemaps;
using UnityEngine.UIElements;

namespace UnityEditor.Tilemaps
{
    [Serializable]
    internal class GridPaintPaletteClipboard : PaintableGrid
    {
        private static List<GridPaintPaletteClipboard> s_Instances;
        public static List<GridPaintPaletteClipboard> instances
        {
            get
            {
                if (s_Instances == null)
                    s_Instances = new List<GridPaintPaletteClipboard>();
                return s_Instances;
            }
        }

        private bool m_DisableOnBrushPicked;
        public event Action onBrushPicked;

        private static readonly string paletteSavedOutsideClipboard = L10n.Tr("Palette Asset {0} was changed outside of the Tile Palette. All changes in the Tile Palette made will be reverted.");

        private bool m_PaletteNeedsSave;
        private const float k_ZoomSpeed = 7f;
        private const float k_MinZoom = 10f; // How many pixels per cell at minimum
        private const float k_MaxZoom = 200f; // How many pixels per cell at maximum
        private const float k_Padding = 1.5f; // How many percentages of window size is the empty padding around the palette content

        private int m_KeyboardPanningID;
        private int m_MousePanningID;
        private Vector2 m_MouseZoomInitialPosition;

        private float k_KeyboardPanningSpeed = 9.0f;

        private Vector3 m_KeyboardPanning;

        private Rect m_GUIRect = new Rect(0, 0, 200, 200);

        public Rect guiRect
        {
            get => m_GUIRect;
            set
            {
                if (m_GUIRect == value)
                    return;
                var oldValue = m_GUIRect;
                m_GUIRect = value;
                OnViewSizeChanged(oldValue, m_GUIRect);
            }
        }

        private VisualElement m_VisualElement;

        public bool activeDragAndDrop { get { return DragAndDrop.objectReferences.Length > 0 && guiRect.Contains(mousePosition); } }

        public bool invalidDragAndDrop => m_HoverData != null && m_HoverData.Count == 0;

        [SerializeField]
        private GridPaletteUtility.GridPaletteType m_FirstUserPaletteType = GridPaletteUtility.GridPaletteType.Rectangle;

        internal GridPaletteUtility.GridPaletteType firstUserPaletteType
        {
            get => m_FirstUserPaletteType;
            set => m_FirstUserPaletteType = value;
        }

        [SerializeField] private bool m_CameraInitializedToBounds;
        [SerializeField] public bool m_CameraPositionSaved;
        [SerializeField] public Vector3 m_CameraPosition;
        [SerializeField] public float m_CameraOrthographicSize;
        [SerializeField] public GridLayout.CellSwizzle m_CameraSwizzleView;

        private Dictionary<Vector2Int, TileDragAndDropHoverData> m_HoverData;
        private bool m_Unlocked;

        private GameObject palette => GridPaintingState.palette;
        private GridBrushBase gridBrush => GridPaintingState.gridBrush;

        private PreviewRenderUtility m_PreviewUtility;
        private PreviewRenderUtility m_PreviewUtilityGizmo;

        internal Vector3 cameraPosition
        {
            get => m_PreviewUtility.camera.transform.position;
            set
            {
                m_PreviewUtility.camera.transform.position = value;
                ClampZoomAndPan();
            }
        }

        internal float cameraSize
        {
            get => m_PreviewUtility.camera.orthographicSize;
            set
            {
                m_PreviewUtility.camera.orthographicSize = value;
                ClampZoomAndPan();
            }
        }

        internal TransparencySortMode cameraTransparencySortMode
        {
            get => m_PreviewUtility.camera.transparencySortMode;
            set => m_PreviewUtility.camera.transparencySortMode = value;
        }

        internal Vector3 cameraTransparencySortAxis
        {
            get => m_PreviewUtility.camera.transparencySortAxis;
            set => m_PreviewUtility.camera.transparencySortAxis = value;
        }

        [SerializeField] private GameObject m_PaletteInstance;

        internal GameObject paletteInstance
        {
            get
            {
                if (m_PaletteInstance == null && palette != null && m_PreviewUtility != null)
                    ResetPreviewInstance();
                return m_PaletteInstance;
            }
        }

        private Tilemap tilemap { get { return paletteInstance != null ? paletteInstance.GetComponentInChildren<Tilemap>() : null; } }
        private Grid grid { get { return paletteInstance != null ? paletteInstance.GetComponent<Grid>() : null; } }
        private Grid prefabGrid { get { return palette != null ? palette.GetComponent<Grid>() : null; } }

        private Mesh m_GridMesh;
        private int m_LastGridHash;
        private Material m_GridMaterial;
        private static readonly PrefColor k_GridColor = new PrefColor("2D/Tile Palette Grid", 255.0f / 255.0f, 255.0f / 255.0f, 255.0f / 255.0f, 25.5f / 255.0f);
        internal static readonly PrefColor tilePaletteBackgroundColor = new PrefColor("2D/Tile Palette Background"
            , 83.0f / 255.0f // Light
            , 83.0f / 255.0f
            , 83.0f / 255.0f
            , 127.0f / 255.0f
            , 43.0f / 255.0f // Dark
            , 43.0f / 255.0f
            , 43.0f / 255.0f
            , 127.0f / 255.0f);
        internal static readonly PrefColor tilePaletteBackgroundEditColor = new PrefColor("2D/Tile Palette Background Edit"
            , 1.0f / 255.0f // Light
            , 35.0f / 255.0f
            , 90.0f / 255.0f
            , 127.0f / 255.0f
            , 1.0f / 255.0f // Dark
            , 35.0f / 255.0f
            , 90.0f / 255.0f
            , 127.0f / 255.0f);

        internal Color backgroundColor => unlocked ? tilePaletteBackgroundEditColor.Color : tilePaletteBackgroundColor.Color;

        private bool m_PaletteUsed; // We mark palette used, when it has been changed in any way during being actively open.
        private Vector2? m_PreviousMousePosition;

        private bool m_DelayedResetPaletteInstance;
        internal void DelayedResetPreviewInstance()
        {
            m_DelayedResetPaletteInstance = true;
        }

        public TileBase activeTile
        {
            get
            {
                if (GridPaintingState.hasActivePick
                    && GridPaintingState.activePick.size == Vector3Int.one
                    && GridPaintingState.defaultBrush != null
                    && GridPaintingState.defaultBrush.cellCount > 0)
                    return GridPaintingState.defaultBrush.cells[0].tile;
                return null;
            }
        }

        private RectInt? m_GameObjectBounds = null;

        private RectInt bounds
        {
            get
            {
                RectInt r = default;
                if (tilemap == null || TilemapIsEmpty(tilemap))
                {
                    if (!isReceivingDragAndDrop)
                        return m_GameObjectBounds.HasValue ? m_GameObjectBounds.Value : r;
                    r = TileDragAndDrop.GetMinMaxRect(m_HoverData.Keys);
                }
                else
                {
                    tilemap.CompressBoundsKeepEditorPreview();
                    r = new RectInt((Vector2Int) tilemap.origin, (Vector2Int) tilemap.size);
                }

                if (m_GameObjectBounds.HasValue)
                {
                    var origin = r.position;
                    var size = r.size;
                    var extent = origin + size;
                    var goOrigin = m_GameObjectBounds.Value.min;
                    var goExtent = m_GameObjectBounds.Value.max;
                    if (goOrigin.x < origin.x)
                        origin.x = goOrigin.x;
                    if (goOrigin.y < origin.y)
                        origin.y = goOrigin.y;
                    if (goExtent.x > extent.x)
                        extent.x = goExtent.x;
                    if (goExtent.y > extent.y)
                        extent.y = goExtent.y;
                    size = extent - origin;
                    r = new RectInt(origin.x, origin.y, size.x, size.y);
                }
                return r;
            }
        }

        private Rect paddedCellBounds
        {
            get
            {
                var aspect = m_GUIRect.width / m_GUIRect.height;
                if (float.IsNaN(aspect) || float.IsInfinity(aspect))
                    return Rect.zero;

                var orthographicSize = m_PreviewUtility.camera.orthographicSize;
                var paddingW = orthographicSize * aspect * k_Padding;
                var paddingH = orthographicSize * k_Padding;
                var rect = bounds;
                var result = new Rect(
                    new Vector2(rect.min.x - paddingW, rect.min.y - paddingH),
                    new Vector2(rect.size.x + paddingW * 2f, rect.size.y + paddingH * 2f));
                return result;
            }
        }

        internal RectInt paddedViewBoundsInt
        {
            get
            {
                var aspect = m_GUIRect.width / m_GUIRect.height;
                if (float.IsNaN(aspect) || float.IsInfinity(aspect))
                    return RectInt.zero;

                var rect = paddedCellBounds;

                var orthographicSize = m_PreviewUtility.camera.orthographicSize;
                var cameraBounds = new Vector3(orthographicSize * aspect * k_Padding, orthographicSize * k_Padding, 0f);
                var camCellBounds = grid.LocalToCellInterpolated(cameraBounds);
                var camSize = camCellBounds * 2f;

                var sizeMax = new Vector2(Math.Max(rect.size.x, camSize.x), Math.Max(rect.size.y, camSize.y));

                var min = Vector3Int.FloorToInt(rect.center - sizeMax * 0.5f);
                var max = Vector3Int.CeilToInt(rect.center + sizeMax * 0.5f);
                return new RectInt(min.x, min.y, max.x - min.x, max.y - min.y);
            }
        }

        // Max area we are ever showing. Depends on the zoom level and content of palette.
        internal Rect paddedViewBounds
        {
            get
            {
                var aspect = m_GUIRect.width / m_GUIRect.height;
                if (float.IsNaN(aspect) || float.IsInfinity(aspect))
                    return Rect.zero;

                var rect = paddedCellBounds;
                var localBounds = grid.GetBoundsLocal(
                    new Vector3(rect.xMin, rect.yMin, 0.0f),
                    new Vector3(rect.size.x, rect.size.y, 0.0f));

                var orthographicSize = m_PreviewUtility.camera.orthographicSize;
                var cameraBounds = new Vector3(orthographicSize * aspect * k_Padding, orthographicSize * k_Padding, 0f) * 2f;

                var size = localBounds.size;
                size.x = Mathf.Max(cameraBounds.x, localBounds.size.x);
                size.y = Mathf.Max(cameraBounds.y, localBounds.size.y);
                localBounds.size = size;

                var result = new Rect(
                    new Vector2(localBounds.min.x, localBounds.min.y),
                    new Vector2(localBounds.size.x, localBounds.size.y));
                return result;
            }
        }

        private GameObject brushTarget
        {
            get
            {
                return (tilemap != null) ? tilemap.gameObject : (grid != null) ? grid.gameObject : null;
            }
        }

        public bool unlocked
        {
            get { return m_Unlocked; }
            set
            {
                if (value == false && m_Unlocked)
                {
                    if (tilemap != null)
                        tilemap.ClearAllEditorPreviewTiles();
                    SavePaletteIfNecessary();
                }
                m_Unlocked = value;
                unlockedChanged?.Invoke(m_Unlocked);
            }
        }
        public event Action<bool> unlockedChanged;

        public bool isReceivingDragAndDrop => m_HoverData != null && m_HoverData.Count > 0;

        public bool isInvalidDragAndDrop => m_HoverData != null && m_HoverData.Count == 0;

        public bool ShowNewEmptyClipboardInfo(GameObject palette)
        {
            if (palette == null)
                return false;

            var tilemap = palette.GetComponentInChildren<Tilemap>(palette);
            if (tilemap == null)
                return false;

            if (unlocked && inEditMode)
                return false;

            if (!TilemapIsEmpty(tilemap))
                return false;

            if (tilemap.transform.childCount > 0)
                return false;

            if (isReceivingDragAndDrop)
                return false;

            if (isInvalidDragAndDrop)
                return false;

            // If user happens to erase the last content of used palette, we don't want to show the new palette info anymore
            if (m_PaletteUsed)
                return false;

            return true;
        }

        public bool isModified { get { return m_PaletteNeedsSave; } }

        internal VisualElement attachedVisualElement
        {
            private get => m_VisualElement;
            set => m_VisualElement = value;
        }

        public void OnBeforePaletteSelectionChanged()
        {
            SavePaletteIfNecessary();
            DestroyPreviewInstance();
            FlushHoverData();
        }

        private void FlushHoverData()
        {
            if (m_HoverData != null)
            {
                m_HoverData.Clear();
                m_HoverData = null;
                if (palette == null)
                    DestroyPreviewInstance();
            }
        }

        public void OnAfterPaletteSelectionChanged()
        {
            m_PaletteUsed = false;
            ResetPreviewInstance();

            if (palette != null)
                ResetPreviewCamera();
        }

        public void SetupPreviewCameraOnInit()
        {
            if (m_CameraPositionSaved)
                LoadSavedCameraPosition();
            else
                ResetPreviewCamera();
        }

        private void LoadSavedCameraPosition()
        {
            m_PreviewUtility.camera.transform.position = m_CameraPosition;
            m_PreviewUtility.camera.orthographicSize = m_CameraOrthographicSize;
            m_PreviewUtility.camera.nearClipPlane = 0.01f;
            m_PreviewUtility.camera.farClipPlane = 100f;
            m_PreviewUtilityGizmo.camera.transform.position = m_CameraPosition;
            m_PreviewUtilityGizmo.camera.orthographicSize = m_CameraOrthographicSize;
            m_PreviewUtilityGizmo.camera.nearClipPlane = 0.01f;
            m_PreviewUtilityGizmo.camera.farClipPlane = 100f;
        }

        private Vector3 GetCameraPositionFromXYZ(Vector3 xyzPosition)
        {
            var position = Grid.Swizzle(m_CameraSwizzleView, xyzPosition);
            position = GetCameraPosition(position);
            return position;
        }

        private Vector3 GetCameraPosition(Vector3 xyzPosition)
        {
            var position = xyzPosition;
            switch (m_CameraSwizzleView)
            {
                case GridLayout.CellSwizzle.XZY:
                    {
                        position.y = 10f;
                    }
                    break;
                case GridLayout.CellSwizzle.YZX:
                    {
                        position.y = -10f;
                    }
                    break;
                case GridLayout.CellSwizzle.ZYX:
                    {
                        position.x = 10f;
                    }
                    break;
                case GridLayout.CellSwizzle.ZXY:
                    {
                        position.x = -10f;
                    }
                    break;
                case GridLayout.CellSwizzle.YXZ:
                    {
                        position.z = 10f;
                    }
                    break;
                case GridLayout.CellSwizzle.XYZ:
                default:
                    {
                        position.z = -10f;
                    }
                    break;
            }
            return position;
        }

        private void ResetPreviewCamera()
        {
            var transform = m_PreviewUtility.camera.transform;

            transform.position = GetCameraPositionFromXYZ(Vector3.zero);
            switch (m_CameraSwizzleView)
            {
                case GridLayout.CellSwizzle.XZY:
                    {
                        transform.rotation = Quaternion.LookRotation(new Vector3(0, -1, 0), new Vector3(0, 0, 1));
                    }
                    break;
                case GridLayout.CellSwizzle.YZX:
                    {
                        transform.rotation = Quaternion.LookRotation(new Vector3(0, 1, 0), new Vector3(1, 0, 0));
                    }
                    break;
                case GridLayout.CellSwizzle.ZXY:
                    {
                        transform.rotation = Quaternion.LookRotation(new Vector3(1, 0, 0), new Vector3(0, 0, 1));
                    }
                    break;
                case GridLayout.CellSwizzle.ZYX:
                    {
                        transform.rotation = Quaternion.LookRotation(new Vector3(-1, 0, 0), new Vector3(0, 1, 0));
                    }
                    break;
                case GridLayout.CellSwizzle.YXZ:
                    {
                        transform.rotation = Quaternion.LookRotation(new Vector3(0, 0, -1), new Vector3(1, 0, 0));
                    }
                    break;
                case GridLayout.CellSwizzle.XYZ:
                default:
                    {
                        transform.rotation = Quaternion.identity;
                    }
                    break;
            }

            m_PreviewUtility.camera.nearClipPlane = 0.01f;
            m_PreviewUtility.camera.farClipPlane = 100f;

            FrameEntirePalette();
        }

        public void InitPreviewUtility()
        {
            m_PreviewUtility = new PreviewRenderUtility(true, true);
            m_PreviewUtility.camera.orthographic = true;
            m_PreviewUtility.camera.orthographicSize = 5f;
            m_PreviewUtility.camera.transform.position = new Vector3(0f, 0f, -10f);
            m_PreviewUtility.ambientColor = new Color(1f, 1f, 1f, 0);

            m_PreviewUtilityGizmo = new PreviewRenderUtility(true, true);
            m_PreviewUtilityGizmo.camera.orthographic = true;
            m_PreviewUtilityGizmo.camera.orthographicSize = 5f;
            m_PreviewUtilityGizmo.camera.transform.position = new Vector3(0f, 0f, -10f);
            m_PreviewUtilityGizmo.ambientColor = new Color(1f, 1f, 1f, 0);
        }

        public void ResetPreviewInstance()
        {
            // Store GridSelection for current Palette Instance
            Stack<int> childPositions = null;
            BoundsInt previousGridSelectionPosition = default;
            if (m_PaletteInstance != null && GridSelection.active && GridSelection.target.transform.IsChildOf(m_PaletteInstance.transform))
            {
                childPositions = new Stack<int>();
                var transform = GridSelection.target.transform;
                while (transform != null && transform != m_PaletteInstance.transform)
                {
                    childPositions.Push(transform.GetSiblingIndex());
                    transform = transform.parent;
                }
                previousGridSelectionPosition = GridSelection.position;
                ClearGridSelection();
            }

            DestroyPreviewInstance();
            if (palette != null)
            {
                m_PaletteInstance = m_PreviewUtility.InstantiatePrefabInScene(palette);

                // Disconnecting prefabs is no longer possible.
                // If performance of overrides on palette palette instance turns out to be a problem.
                // unpack the prefab instance here, and overwrite the prefab later instead of reconnecting.
                PrefabUtility.UnpackPrefabInstance(m_PaletteInstance, PrefabUnpackMode.OutermostRoot, InteractionMode.AutomatedAction);

                EditorUtility.InitInstantiatedPreviewRecursive(m_PaletteInstance);
                m_PaletteInstance.transform.position = new Vector3(0, 0, 0);
                m_PaletteInstance.transform.rotation = Quaternion.identity;
                m_PaletteInstance.transform.localScale = Vector3.one;

                var paletteAsset = GridPaletteUtility.GetGridPaletteFromPaletteAsset(palette);
                if (paletteAsset != null)
                {
                    // Handle Cell Sizing for Palette
                    var paletteGrid = m_PaletteInstance.GetComponent<Grid>();
                    if (paletteAsset.cellSizing == GridPalette.CellSizing.Automatic)
                    {
                        if (paletteGrid != null)
                        {
                            paletteGrid.cellSize = GridPaletteUtility.CalculateAutoCellSize(paletteGrid, paletteGrid.cellSize);
                        }
                        else
                        {
                            Debug.LogWarning("Grid component not found from: " + palette.name);
                        }
                    }

                    // Handle Transparency Sort Settings
                    m_PreviewUtility.camera.transparencySortMode = paletteAsset.transparencySortMode;
                    m_PreviewUtility.camera.transparencySortAxis = paletteAsset.transparencySortAxis;

                    // Handle Camera View for Grid
                    m_CameraSwizzleView = GridLayout.CellSwizzle.XYZ;
                    if (paletteGrid != null && (tilemap == null || tilemap.orientation != Tilemap.Orientation.XY))
                    {
                        // Set SwizzleView only if Tilemap is not oriented to XY
                        m_CameraSwizzleView = paletteGrid.cellSwizzle;
                    }
                }
                else
                {
                    Debug.LogWarning("GridPalette subasset not found from: " + palette.name);
                    m_PreviewUtility.camera.transparencySortMode = TransparencySortMode.Default;
                    m_PreviewUtility.camera.transparencySortAxis = new Vector3(0f, 0f, 1f);
                }

                foreach (var transform in m_PaletteInstance.GetComponentsInChildren<Transform>())
                    transform.gameObject.hideFlags = HideFlags.HideAndDontSave;

                // Show all renderers from Palettes from previous versions
                var goBounds = SetEnableRenderersAndGetBounds(m_PaletteInstance, true);
                if (float.IsNormal(goBounds.x) && float.IsNormal(goBounds.y))
                {
                    m_GameObjectBounds = new RectInt(Mathf.FloorToInt(goBounds.x)
                        , Mathf.FloorToInt(goBounds.y)
                        , Mathf.CeilToInt(goBounds.width)
                        , Mathf.CeilToInt(goBounds.height));
                }
                else
                {
                    m_GameObjectBounds = null;
                }

                // Update preview Grid Mesh for new palette instance
                ResetPreviewGridMesh();

                // Restore GridSelection for new palette instance
                if (childPositions != null)
                {
                    var transform = m_PaletteInstance.transform;
                    while (childPositions.Count > 0)
                    {
                        var siblingIndex = childPositions.Pop();
                        if (siblingIndex < transform.childCount)
                            transform = transform.GetChild(siblingIndex);
                    }
                    GridSelection.Select(transform.gameObject, previousGridSelectionPosition);
                }
            }
            m_DelayedResetPaletteInstance = false;
        }

        internal static Rect SetEnableRenderersAndGetBounds(GameObject go, bool enabled)
        {
            var origin = Vector2.positiveInfinity;
            var extents = Vector2.negativeInfinity;
            foreach (Renderer renderer in go.GetComponentsInChildren<Renderer>())
            {
                renderer.enabled = enabled;
                if (renderer is not TilemapRenderer)
                {
                    var rbounds = renderer.bounds;
                    if (rbounds.min.x < origin.x)
                        origin.x = rbounds.min.x;
                    if (rbounds.min.y < origin.y)
                        origin.y = rbounds.min.y;
                    if (rbounds.max.x > extents.x)
                        extents.x = rbounds.max.x;
                    if (rbounds.min.x < extents.x)
                        extents.y = rbounds.max.y;
                }
            }
            return new Rect(origin, extents - origin);
        }

        public void CreateTemporaryPreviewInstance()
        {
            if (palette != null)
                return;

            m_DelayedResetPaletteInstance = false;
            DestroyPreviewInstance();

            m_PaletteInstance = new GameObject("First User Palette", typeof(Grid), typeof(Tilemap),
                typeof(TilemapRenderer));
            m_PreviewUtility.camera.transparencySortMode = TransparencySortMode.Default;
            m_PreviewUtility.camera.transparencySortAxis = new Vector3(0f, 0f, 1f);
            var paletteGrid = m_PaletteInstance.GetComponent<Grid>();

            switch (m_FirstUserPaletteType)
            {
                case GridPaletteUtility.GridPaletteType.Rectangle:
                {
                    paletteGrid.cellLayout = GridLayout.CellLayout.Rectangle;
                    paletteGrid.cellSize = new Vector3(1, 1, 0);
                    break;
                }
                case GridPaletteUtility.GridPaletteType.HexagonalPointTop:
                {
                    paletteGrid.cellLayout = GridLayout.CellLayout.Hexagon;
                    paletteGrid.cellSize = new Vector3(1, 1, 0);
                    break;
                }
                case GridPaletteUtility.GridPaletteType.HexagonalFlatTop:
                {
                    paletteGrid.cellLayout = GridLayout.CellLayout.Hexagon;
                    paletteGrid.cellSize = new Vector3(1, 1, 0);
                    paletteGrid.cellSwizzle = GridLayout.CellSwizzle.YXZ;
                    break;
                }
                case GridPaletteUtility.GridPaletteType.Isometric:
                {
                    paletteGrid.cellLayout = GridLayout.CellLayout.Isometric;
                    paletteGrid.cellSize = new Vector3(1, 0.5f, 1);
                    break;
                }
                case GridPaletteUtility.GridPaletteType.IsometricZAsY:
                {
                    paletteGrid.cellLayout = GridLayout.CellLayout.IsometricZAsY;
                    paletteGrid.cellSize = new Vector3(1, 0.5f, 1);
                    m_PreviewUtility.camera.transparencySortMode  = TransparencySortMode.CustomAxis;
                    m_PreviewUtility.camera.transparencySortAxis  = new Vector3(0f, 1f, -0.25f);
                    break;
                }
            }

            EditorUtility.InitInstantiatedPreviewRecursive(m_PaletteInstance);
            m_PaletteInstance.transform.position = new Vector3(0, 0, 0);
            m_PaletteInstance.transform.rotation = Quaternion.identity;
            m_PaletteInstance.transform.localScale = Vector3.one;

            foreach (var transform in m_PaletteInstance.GetComponentsInChildren<Transform>())
                transform.gameObject.hideFlags = HideFlags.HideAndDontSave;

            // Show all renderers from Palettes from previous versions
            PreviewRenderUtility.SetEnabledRecursive(m_PaletteInstance, true);

            // Update preview Grid Mesh for new palette instance
            ResetPreviewGridMesh();
        }

        public void DestroyPreviewInstance()
        {
            if (m_PaletteInstance != null)
            {
                Undo.ClearUndo(m_PaletteInstance);
                if (GridSelection.active && GridSelection.target == tilemap.gameObject)
                {
                    GridSelection.TransferToStandalone(palette);
                }
                else
                {
                    DestroyImmediate(m_PaletteInstance);
                }
                m_PaletteInstance = null;
            }
        }

        private void ResetPreviewGridMesh()
        {
            if (m_GridMesh != null)
            {
                DestroyImmediate(m_GridMesh);
                m_GridMesh = null;
            }
            m_GridMaterial = null;
        }

        public void FrameEntirePalette()
        {
            Frame(bounds);
        }

        private void Frame(RectInt rect)
        {
            if (grid == null)
                return;

            var position = grid.CellToLocalInterpolated(new Vector3(rect.center.x, rect.center.y, 0));
            position = GetCameraPosition(position);

            var height = (grid.CellToLocal(new Vector3Int(0, rect.yMax, 0)) - grid.CellToLocal(new Vector3Int(0, rect.yMin, 0))).magnitude;
            var width = (grid.CellToLocal(new Vector3Int(rect.xMax, 0, 0)) - grid.CellToLocal(new Vector3Int(rect.xMin, 0, 0))).magnitude;

            var cellSize = grid.cellSize;
            width += cellSize.x;
            height += cellSize.y;

            var guiAspect = m_GUIRect.width / m_GUIRect.height;
            var contentAspect = width / height;

            m_PreviewUtility.camera.transform.position = position;
            m_PreviewUtility.camera.orthographicSize = (guiAspect > contentAspect ? height : width / guiAspect) / 2f;

            ClampZoomAndPan();
        }

        private void RefreshAllTiles()
        {
            if (tilemap != null)
                tilemap.RefreshAllTiles();
        }

        protected override void OnEnable()
        {
            base.OnEnable();

            instances.Add(this);

            EditorApplication.editorApplicationQuit += EditorApplicationQuit;
            PrefabUtility.prefabInstanceUpdated += PrefabInstanceUpdated;
            Undo.undoRedoPerformed += UndoRedoPerformed;

            m_KeyboardPanningID = GUIUtility.GetPermanentControlID();
            m_MousePanningID = GUIUtility.GetPermanentControlID();

            InitPreviewUtility();
            ResetPreviewInstance();
            SetupPreviewCameraOnInit();
        }

        protected override void OnDisable()
        {
            SavePaletteIfNecessary();
            unlocked = false;
            DestroyPreviewInstance();

            if (m_PreviewUtility != null && m_PreviewUtility.camera != null)
            {
                // Save Preview camera coordinates
                m_CameraPosition = m_PreviewUtility.camera.transform.position;
                m_CameraOrthographicSize = m_PreviewUtility.camera.orthographicSize;
                m_CameraPositionSaved = true;
            }
            ResetPreviewGridMesh();

            if (m_PreviewUtility != null)
                m_PreviewUtility.Cleanup();
            m_PreviewUtility = null;
            if (m_PreviewUtilityGizmo != null)
                m_PreviewUtilityGizmo.Cleanup();
            m_PreviewUtilityGizmo = null;

            Undo.undoRedoPerformed -= UndoRedoPerformed;
            PrefabUtility.prefabInstanceUpdated -= PrefabInstanceUpdated;
            EditorApplication.editorApplicationQuit -= EditorApplicationQuit;

            GridPaintingState.RemoveActiveGrid(this);
            instances.Remove(this);

            base.OnDisable();
        }

        private void OnViewSizeChanged(Rect oldSize, Rect newSize)
        {
            if (Mathf.Approximately(oldSize.height * oldSize.width * newSize.height * newSize.width, 0f))
                return;

            ClampZoomAndPan();
        }

        private void EditorApplicationQuit()
        {
            SavePaletteIfNecessary();
        }

        private void UndoRedoPerformed()
        {
            if (!unlocked)
                return;

            m_PaletteNeedsSave = true;
            RefreshAllTiles();
            Repaint();
        }

        private void PrefabInstanceUpdated(GameObject updatedPrefab)
        {
            // case 947462: Reset the palette instance after its prefab has been updated as it could have been changed
            if (m_PaletteInstance != null && PrefabUtility.GetCorrespondingObjectFromSource(updatedPrefab) == palette && !GridPaintingState.savingPalette)
            {
                m_PaletteNeedsSave = true;
                RefreshAllTiles();
                Repaint();
            }
        }

        private bool IsGUIRectValid()
        {
            if (guiRect.width <= 0f
                || guiRect.height <= 0f
                || float.IsNaN(guiRect.width)
                || float.IsNaN(guiRect.height))
                return false;
            return true;
        }

        public Texture HandleIMGUI()
        {
            if (!IsGUIRectValid())
                return null;

            HandleUtility.BeginHandles();
            m_PreviewUtilityGizmo.BeginPreview(guiRect, null);
            Handles.SetCamera(m_PreviewUtility.camera);
            CallOnSceneGUI();
            m_PreviewUtilityGizmo.EndPreview();
            HandleUtility.EndHandles();
            return Event.current.type == EventType.Repaint ? RenderTexture() : null;
        }

        public void HandleExecuteCommandEvent(ExecuteCommandEvent evt)
        {
            if (evt.commandName == EventCommandNames.FrameSelected)
            {
                if (GridPaintingState.hasActivePick)
                {
                    var rect = new RectInt(GridPaintingState.activePick.x, GridPaintingState.activePick.y,
                        GridPaintingState.activePick.size.x, GridPaintingState.activePick.size.y);
                    Frame(rect);
                }
                else
                    FrameEntirePalette();
                evt.StopPropagation();
            }
        }

        public void HandleValidateCommandEvent(ValidateCommandEvent evt)
        {
            if (evt.commandName == EventCommandNames.FrameSelected)
            {
                evt.StopPropagation();
            }
        }

        public void HandlePointerDownEvent(PointerDownEvent evt, int button, bool alt, bool control, Vector3 localMousePosition)
        {
            if (grid != null)
                UpdateMouseGridPositionUIToolkit(localMousePosition);

            if (m_MarqueeStart != null)
            {
                m_MarqueeType = MarqueeType.None;
                m_MarqueeStart = null;
                GUIUtility.hotControl = 0;
            }

            // Mouse Panning
            if (MousePanningEvent(button, alt) && guiRect.Contains(localMousePosition) && GUIUtility.hotControl == 0)
            {
                GUIUtility.hotControl = m_MousePanningID;
                m_MouseZoomInitialPosition = localMousePosition;
                evt.StopPropagation();
            }

            // Brush Picking
            if (IsPickingEvent(evt) && !isHotControl)
            {
                m_TypeBeforeExecution = typeof(PaintTool);
                if (inEditMode && !TilemapEditorTool.IsActive(typeof(PickingTool)))
                {
                    m_TypeBeforeExecution = ToolManager.activeToolType;
                }
                TilemapEditorTool.SetActiveEditorTool(typeof(PickingTool));

                OnBrushPickCancelled();
                m_MarqueeStart = mouseGridPosition;
                m_MarqueeType = MarqueeType.Pick;
                evt.StopPropagation();
                GUI.changed = true;
                GUIUtility.hotControl = m_PermanentControlID;
                OnBrushPickStarted();
            }

            if (!inEditMode)
                return;

            // Brush Select
            if (IsSelectingEvent(evt))
            {
                if (TilemapEditorTool.IsActive(typeof(MoveTool)) && control)
                    TilemapEditorTool.SetActiveEditorTool(typeof(SelectTool));

                m_PreviousMove = null;
                m_MarqueeStart = mouseGridPosition;
                m_MarqueeType = MarqueeType.Select;

                GUIUtility.hotControl = m_PermanentControlID;
                evt.StopPropagation();
            }

            if (!unlocked)
                return;

            // Brush Painting
            if (IsPaintingEvent(evt) || IsErasingEvent(evt))
            {
                RegisterUndo();
                GUIUtility.hotControl = m_PermanentControlID;
                executing = true;
                if (ToolManager.activeToolType != null &&  ToolManager.activeToolType.IsSubclassOf(typeof(TilemapEditorTool)))
                    m_TypeBeforeExecution = ToolManager.activeToolType;
                var position = new Vector3Int(mouseGridPosition.x, mouseGridPosition.y, zPosition);
                if (IsErasingEvent(evt))
                {
                    if (!TilemapEditorTool.IsActive(typeof(EraseTool)))
                        TilemapEditorTool.SetActiveEditorTool(typeof(EraseTool));
                    Erase(position);
                }
                else
                {
                    if (!TilemapEditorTool.IsActive(typeof(PaintTool)))
                        TilemapEditorTool.SetActiveEditorTool(typeof(PaintTool));
                    Paint(position);
                }
                ResetPreviousMousePositionToCurrentPosition();
                evt.StopPropagation();
                GUI.changed = true;
            }

            // Brush Move
            if (IsMoveEvent(evt))
            {
                RegisterUndo();
                var mouse3D = new Vector3Int(mouseGridPosition.x, mouseGridPosition.y, GridSelection.position.zMin);
                if (GridSelection.active && GridSelection.position.Contains(mouse3D))
                {
                    GUIUtility.hotControl = m_PermanentControlID;
                    executing = true;
                    m_MarqueeStart = null;
                    m_MarqueeType = MarqueeType.None;
                    m_PreviousMove = mouseGridPosition;
                    MoveStart(GridSelection.position);
                }
                evt.StopPropagation();
            }

            // Brush FloodFill
            if (TilemapEditorTool.IsActive(typeof(FillTool)) && GridPaintingState.gridBrush != null &&
                ValidateFloodFillPosition(new Vector3Int(mouseGridPosition.x, mouseGridPosition.y, 0)))
            {
                if (button == 0 && !alt)
                {
                    GUIUtility.hotControl = m_PermanentControlID;
                    GUI.changed = true;
                    executing = true;
                    evt.StopPropagation();
                }
            }

            // Brush Box
            if (button == 0 && !alt && TilemapEditorTool.IsActive(typeof(BoxTool)))
            {
                m_MarqueeStart = mouseGridPosition;
                m_MarqueeType = MarqueeType.Box;
                GUI.changed = true;
                executing = true;
                evt.StopPropagation();
                GUIUtility.hotControl = m_PermanentControlID;
            }

            // Brush Custom Tool
            if (TilemapEditorTool.IsCustomTilemapEditorToolActive())
            {
                var activeTool = EditorToolManager.activeTool as TilemapEditorTool;
                var executed = CustomTool(isHotControl, activeTool, new Vector3Int(mouseGridPosition.x, mouseGridPosition.y, zPosition));
                if (executed != executing)
                {
                    GUIUtility.hotControl = executed ? m_PermanentControlID : 0;
                    executing = executed;
                    GUI.changed = true;
                    evt.StopPropagation();
                }
                else if (executing)
                {
                    GUI.changed = true;
                    evt.StopPropagation();
                }
            }
        }

        public void HandlePointerMoveEvent(PointerMoveEvent evt, int button, bool alt, Vector3 localMousePosition, Vector3 deltaPosition)
        {
            if (grid == null)
                return;

            var isButton0 = (evt.pressedButtons & 1) > 0;

            UpdateMouseGridPositionUIToolkit(localMousePosition, true);

            // Mouse Panning
            if (GUIUtility.hotControl == m_MousePanningID)
            {
                if (alt && button == 1)
                {
                    HandleWheelEvent(deltaPosition, m_MouseZoomInitialPosition, evt.shiftKey);
                }
                else
                {
                    var delta = new Vector3(-deltaPosition.x, deltaPosition.y, 0f) / LocalToScreenRatio();
                    m_PreviewUtility.camera.transform.Translate(delta);
                    ClampZoomAndPan();
                }
                evt.StopPropagation();
            }

            // Brush Pick
            if (isHotControl && m_MarqueeStart.HasValue && m_MarqueeType == MarqueeType.Pick && IsPickingEvent(evt))
            {
                var rect = GridEditorUtility.GetMarqueeRect(m_MarqueeStart.Value, mouseGridPosition);
                OnBrushPickDragged(new BoundsInt(new Vector3Int(rect.xMin, rect.yMin, zPosition), new Vector3Int(rect.size.x, rect.size.y, 1)));
                evt.StopPropagation();
                GUI.changed = true;
            }

            if (!inEditMode || !unlocked)
                return;

            // Brush Painting
            if (IsPaintingEvent(evt) || IsErasingEvent(evt))
            {
                if (isHotControl && executing && mouseGridPositionChanged)
                {
                    var points = GridEditorUtility.GetPointsOnLine(m_PreviousMouseGridPosition, mouseGridPosition);

                    if (!evt.shiftKey && !TilemapEditorTool.IsActive(typeof(PaintTool)) && m_TypeBeforeExecution == typeof(PaintTool))
                        TilemapEditorTool.SetActiveEditorTool(typeof(PaintTool));
                    else if (evt.shiftKey && TilemapEditorTool.IsActive(typeof(PaintTool)))
                        TilemapEditorTool.SetActiveEditorTool(typeof(EraseTool));

                    foreach (var point in points)
                    {
                        var position = new Vector3Int(point.x, point.y, zPosition);
                        if (IsErasingEvent(evt))
                            Erase(position);
                        else
                            Paint(position);
                    }
                    ResetPreviousMousePositionToCurrentPosition();
                    evt.StopPropagation();
                    GUI.changed = true;
                }
            }

            // Brush Move
            if (isButton0 && TilemapEditorTool.IsActive(typeof(MoveTool)) && isHotControl)
            {
                if (m_MouseGridPositionChanged && m_PreviousMove.HasValue)
                {
                    executing = true;
                    var previousRect = GridSelection.position;
                    var previousBounds = new BoundsInt(new Vector3Int(previousRect.xMin, previousRect.yMin, GridSelection.position.zMin), new Vector3Int(previousRect.size.x, previousRect.size.y, 1));

                    var direction = mouseGridPosition - m_PreviousMove.Value;
                    var pos = GridSelection.position;
                    pos.position = new Vector3Int(pos.x + direction.x, pos.y + direction.y, pos.z);
                    GridSelection.position = pos;
                    Move(previousBounds, pos);
                    m_PreviousMove = mouseGridPosition;
                    evt.StopPropagation();
                }
            }
        }

        public void HandlePointerUpEvent(PointerUpEvent evt)
        {
            // Mouse Panning
            if (GUIUtility.hotControl == m_MousePanningID)
            {
                ClampZoomAndPan();
                GUIUtility.hotControl = 0;
                evt.StopPropagation();
            }
            // Brush Pick
            if (isHotControl && m_MarqueeStart.HasValue && m_MarqueeType == MarqueeType.Pick && IsPickingEvent(evt))
            {
                // Check if event only occurred in the PaintableGrid window as evt.type will filter for this
                if (IsMouseUpInWindow() && m_MarqueeType == MarqueeType.Pick)
                {
                    var rect = GridEditorUtility.GetMarqueeRect(m_MarqueeStart.Value, mouseGridPosition);
                    var pivot = GetMarqueePivot(m_MarqueeStart.Value, mouseGridPosition);
                    PickBrush(new BoundsInt(new Vector3Int(rect.xMin, rect.yMin, zPosition), new Vector3Int(rect.size.x, rect.size.y, 1)), new Vector3Int(pivot.x, pivot.y, 0));

                    if (inEditMode && ToolManager.activeToolType != m_TypeBeforeExecution)
                    {
                        if (PickingIsDefaultTool()
                            && (m_TypeBeforeExecution == typeof(EraseTool)
                                || m_TypeBeforeExecution == typeof(MoveTool)))
                        {
                            // If Picking is default, change to a Paint Tool to facilitate editing if previous tool does not allow for painting
                            TilemapEditorTool.SetActiveEditorTool(typeof(PaintTool));
                        }
                        else
                        {
                            TilemapEditorTool.SetActiveEditorTool(m_TypeBeforeExecution);
                        }
                    }

                    GridPaintingState.ActiveGridBrushAssetChanged();
                    evt.StopPropagation();
                    GUI.changed = true;
                }
                else
                // Event occurred outside of PaintableGrid window, cancel the pick event
                {
                    OnBrushPickCancelled();
                }
                m_MarqueeType = MarqueeType.None;
                m_MarqueeStart = null;
                GUIUtility.hotControl = 0;
                InspectorWindow.RepaintAllInspectors();
            }

            if (!inEditMode)
                return;

            // Brush Select
            if (IsSelectingEvent(evt) && m_MarqueeStart.HasValue && isHotControl)
            {
                // Check if event only occurred in the PaintableGrid window as evt.type will filter for this
                if (IsMouseUpInWindow() && m_MarqueeType == MarqueeType.Select)
                {
                    var rect = GridEditorUtility.GetMarqueeRect(m_MarqueeStart.Value, mouseGridPosition);
                    Select(new BoundsInt(new Vector3Int(rect.xMin, rect.yMin, zPosition), new Vector3Int(rect.size.x, rect.size.y, 1)));
                    evt.StopPropagation();
                }
                if (evt.ctrlKey)
                    TilemapEditorTool.SetActiveEditorTool(typeof(MoveTool));
                m_MarqueeStart = null;
                m_MarqueeType = MarqueeType.None;
                InspectorWindow.RepaintAllInspectors();
                GUIUtility.hotControl = 0;
            }

            if (!unlocked)
                return;

            // Brush Painting
            if (IsPaintingEvent(evt) || IsErasingEvent(evt))
            {
                executing = false;
                if (isHotControl)
                {
                    if (!TilemapEditorTool.IsActive(typeof(PaintTool)) && m_TypeBeforeExecution == typeof(PaintTool))
                    {
                        TilemapEditorTool.SetActiveEditorTool(typeof(PaintTool));
                    }

                    evt.StopPropagation();
                    GUI.changed = true;
                    GUIUtility.hotControl = 0;
                }
            }

            // Brush Move
            if (IsMouseUpInWindow() && IsMoveEvent(evt) && m_PreviousMove.HasValue && isHotControl)
            {
                m_PreviousMove = null;
                MoveEnd(GridSelection.position);
                executing = false;
                evt.StopPropagation();
                GUIUtility.hotControl = 0;
            }

            // Brush FloodFill
            if (TilemapEditorTool.IsActive(typeof(FillTool)) && GridPaintingState.gridBrush != null && ValidateFloodFillPosition(new Vector3Int(mouseGridPosition.x, mouseGridPosition.y, 0)))
            {
                if (IsMouseUpInWindow() && evt.button == 0 && isHotControl)
                {
                    RegisterUndo();
                    FloodFill(new Vector3Int(mouseGridPosition.x, mouseGridPosition.y, zPosition));
                    executing = false;
                    GUI.changed = true;
                    evt.StopPropagation();
                    GUIUtility.hotControl = 0;
                }
            }

            // Brush Box
            if (IsMouseUpInWindow() && evt.button == 0 && TilemapEditorTool.IsActive(typeof(BoxTool)))
            {
                if (isHotControl && m_MarqueeStart.HasValue)
                {
                    RegisterUndo();
                    var rect = GridEditorUtility.GetMarqueeRect(m_MarqueeStart.Value, mouseGridPosition);
                    if (evt.shiftKey)
                        BoxErase(new BoundsInt(rect.x, rect.y, zPosition, rect.size.x, rect.size.y, 1));
                    else
                        BoxFill(new BoundsInt(rect.x, rect.y, zPosition, rect.size.x, rect.size.y, 1));
                    executing = false;
                    GUI.changed = true;
                    evt.StopPropagation();
                    GUIUtility.hotControl = 0;
                }
                m_MarqueeStart = null;
                m_MarqueeType = MarqueeType.None;
            }

            // Brush Custom Tool
            if (TilemapEditorTool.IsCustomTilemapEditorToolActive())
            {
                var activeTool = EditorToolManager.activeTool as TilemapEditorTool;
                var executed = CustomTool(isHotControl, activeTool, new Vector3Int(mouseGridPosition.x, mouseGridPosition.y, zPosition));
                if (executed != executing)
                {
                    GUIUtility.hotControl = executed ? m_PermanentControlID : 0;
                    executing = executed;
                    GUI.changed = true;
                    evt.StopPropagation();
                }
                else if (executing)
                {
                    GUI.changed = true;
                    evt.StopPropagation();
                }
            }
        }

        public void HandlePointerEnterEvent(PointerEnterEvent evt)
        {
            if (m_PreviousMousePosition.HasValue && !guiRect.Contains(m_PreviousMousePosition.Value) || !m_PreviousMousePosition.HasValue)
            {
                if (GridPaintingState.activeGrid != this)
                {
                    GridPaintingState.AddActiveGrid(this);
                }
                if (GridPaintingState.activeBrushEditor != null)
                {
                    GridPaintingState.activeBrushEditor.OnMouseEnter();
                }
            }
        }

        public void HandlePointerLeaveEvent(PointerLeaveEvent evt)
        {
            // Mouse Pick
            if (isHotControl
                && m_MarqueeStart.HasValue && m_MarqueeType == MarqueeType.Pick
                && IsPickingEvent(evt))
            {
                // Event occurred outside of PaintableGrid window, cancel the pick event
                OnBrushPickCancelled();
                InspectorWindow.RepaintAllInspectors();
            }

            // Mouse Leave
            var gridActive = GridPaintingState.activeGrid == this;
            GridPaintingState.RemoveActiveGrid(this);
            if (gridActive)
            {
                if (GridPaintingState.activeBrushEditor != null)
                {
                    GridPaintingState.activeBrushEditor.OnMouseLeave();
                    Repaint();
                }
                FlushHoverData();
            }
        }

        public void HandleKeyDownEvent(KeyDownEvent evt)
        {
            var keyCode = evt.keyCode;
            var shift = evt.shiftKey;
            if ((GUIUtility.hotControl == 0 || GUIUtility.hotControl == m_KeyboardPanningID))
            {
                // Keyboard Panning
                if (!shift)
                {
                    switch (keyCode)
                    {
                        case KeyCode.LeftArrow:
                            m_KeyboardPanning.x = -k_KeyboardPanningSpeed / LocalToScreenRatio();
                            GUIUtility.hotControl = m_KeyboardPanningID;
                            break;
                        case KeyCode.RightArrow:
                            m_KeyboardPanning.x = k_KeyboardPanningSpeed / LocalToScreenRatio();
                            GUIUtility.hotControl = m_KeyboardPanningID;
                            break;
                        case KeyCode.UpArrow:
                            m_KeyboardPanning.y = k_KeyboardPanningSpeed / LocalToScreenRatio();
                            GUIUtility.hotControl = m_KeyboardPanningID;
                            break;
                        case KeyCode.DownArrow:
                            m_KeyboardPanning.y = -k_KeyboardPanningSpeed / LocalToScreenRatio();
                            GUIUtility.hotControl = m_KeyboardPanningID;
                            break;
                    }
                    if (GUIUtility.hotControl == m_KeyboardPanningID)
                    {
                        m_PreviewUtility.camera.transform.Translate(m_KeyboardPanning);
                        ClampZoomAndPan();
                        Repaint();
                        evt.StopPropagation();
                    }
                }

                // Brush Pick selection
                if (shift && GridPaintingState.hasActivePick)
                {
                    var delta = Vector3Int.zero;
                    switch (keyCode)
                    {
                        case KeyCode.LeftArrow:
                            delta = Vector3Int.left;
                            break;
                        case KeyCode.RightArrow:
                            delta = Vector3Int.right;
                            break;
                        case KeyCode.UpArrow:
                            delta = Vector3Int.up;
                            break;
                        case KeyCode.DownArrow:
                            delta = Vector3Int.down;
                            break;
                    }

                    if (delta != Vector3Int.zero)
                    {
                        m_DisableOnBrushPicked = true;
                        PickBrush(new BoundsInt(GridPaintingState.activePick.position + delta, GridPaintingState.activePick.size),
                            GridPaintingState.activePivot);
                        GridPaintingState.ActiveGridBrushAssetChanged();
                        m_DisableOnBrushPicked = false;
                        evt.StopPropagation();
                    }
                }
            }

            // Drag Escape
            if (m_HoverData != null && (keyCode == KeyCode.Escape))
            {
                DragAndDrop.visualMode = DragAndDropVisualMode.None;
                FlushHoverData();
                evt.StopPropagation();
            }

            // Grid Selection
            if (keyCode == KeyCode.Escape
                && !m_MarqueeStart.HasValue
                && !m_PreviousMove.HasValue)
            {
                ClearGridSelection();
                evt.StopPropagation();
            }

            // Other
            if (!evt.isPropagationStopped)
            {
                CallOnSceneGUI();
            }
        }

        public void HandleKeyUpEvent()
        {
            if (GUIUtility.hotControl == m_KeyboardPanningID)
            {
                m_KeyboardPanning = Vector3.zero;
                GUIUtility.hotControl = 0;
            }
        }

        public void HandleWheelEvent(Vector3 delta, Vector2 currentMousePosition, bool shift)
        {
            var zoomDelta = HandleUtility.niceMouseDeltaZoom * (shift ? -9 : -3) * k_ZoomSpeed;
            var camera = m_PreviewUtility.camera;
            var oldLocalPos = ScreenToLocal(currentMousePosition);
            camera.orthographicSize = Mathf.Max(.0001f, camera.orthographicSize * (1 + zoomDelta * .001f));
            ClampZoomAndPan();
            var newLocalPos = ScreenToLocal(currentMousePosition);
            var localDelta = newLocalPos - oldLocalPos;
            camera.transform.position -= (Vector3) localDelta;
            ClampZoomAndPan();
            EditorGUIUtility.AddCursorRect(guiRect, MouseCursor.Zoom);
        }

        public void HandleDragEnterEvent(DragEnterEvent evt)
        {
            if (palette != null && !GridPaintingState.isPaletteEditable)
                return;

            if (m_HoverData == null)
            {
                var sheets = TileDragAndDrop.GetValidSpritesheets(DragAndDrop.objectReferences);
                var sprites = TileDragAndDrop.GetValidSingleSprites(DragAndDrop.objectReferences);
                var tiles = TileDragAndDrop.GetValidTiles(DragAndDrop.objectReferences);
                var gos = TileDragAndDrop.GetValidGameObjects(DragAndDrop.objectReferences);

                var noPalette = tilemap == null || palette == null;
                if (!noPalette)
                {
                    TileDragAndDrop.FilterForValidGameObjectsForPrefab(palette, gos);
                }

                var targetLayout = !noPalette ? tilemap.cellLayout : GridPaletteUtility.GetCellLayoutFromGridPaletteType(m_FirstUserPaletteType);
                m_HoverData = TileDragAndDrop.CreateHoverData(sheets, sprites, tiles, gos, targetLayout);
                if (m_HoverData is { Count: > 0 })
                {
                    if (noPalette)
                        CreateTemporaryPreviewInstance();
                }
            }

            if (m_HoverData != null)
            {
                ClampZoomAndPan();
                DragAndDrop.visualMode = m_HoverData.Count > 0 ? DragAndDropVisualMode.Copy : DragAndDropVisualMode.Rejected;
                evt.StopPropagation();
                GUI.changed = true;
            }
        }

        public void HandleDragUpdatedEvent(DragUpdatedEvent evt)
        {
            if (grid == null)
                return;

            UpdateMouseGridPositionUIToolkit(evt.localMousePosition, true);
        }

        public void HandleDragPerformEvent(DragPerformEvent evt)
        {
            var performed = DoDragPerform();
            if (performed)
                evt.StopPropagation();
        }

        public void HandleDragLeaveEvent(DragLeaveEvent evt)
        {
            if (m_HoverData != null)
            {
                DragAndDrop.visualMode = DragAndDropVisualMode.None;
                FlushHoverData();
                evt.StopPropagation();
            }
        }

        public void HandleDragExitedEvent(DragExitedEvent evt)
        {
            if (m_HoverData != null)
            {
                DragAndDrop.visualMode = DragAndDropVisualMode.None;
                FlushHoverData();
                evt.StopPropagation();
            }
        }

        private static bool MousePanningEvent(int button, bool alt)
        {
            return (button == 0 && alt || button > 0);
        }

        private void ClampZoomAndPan()
        {
            if (grid == null)
                return;

            var pixelsPerCell = grid.cellSize.y * LocalToScreenRatio();
            if (pixelsPerCell < k_MinZoom)
                m_PreviewUtility.camera.orthographicSize = (grid.cellSize.y * guiRect.height) / (k_MinZoom * 2f);
            else if (pixelsPerCell > k_MaxZoom)
                m_PreviewUtility.camera.orthographicSize = (grid.cellSize.y * guiRect.height) / (k_MaxZoom * 2f);

            var cam = m_PreviewUtility.camera;
            var cameraOrthographicSize = cam.orthographicSize;
            var r = paddedViewBounds;

            var camPos = cam.transform.position;
            var camLimit = Grid.Swizzle(m_CameraSwizzleView, new Vector3(cameraOrthographicSize * (guiRect.width / guiRect.height), cameraOrthographicSize));
            var camMin = camPos - camLimit;
            var camMax = camPos + camLimit;
            var rMin = Grid.Swizzle(m_CameraSwizzleView, r.min);
            var rMax = Grid.Swizzle(m_CameraSwizzleView, r.max);

            if (m_CameraSwizzleView != GridLayout.CellSwizzle.ZXY && m_CameraSwizzleView != GridLayout.CellSwizzle.ZYX)
            {
                if (camMin.x < rMin.x)
                {
                    camPos += new Vector3(rMin.x - camMin.x, 0f, 0f);
                }
                if (camMax.x > rMax.x)
                {
                    camPos += new Vector3(rMax.x - camMax.x, 0f, 0f);
                }
            }

            if (m_CameraSwizzleView != GridLayout.CellSwizzle.XZY && m_CameraSwizzleView != GridLayout.CellSwizzle.YZX)
            {
                if (camMin.y < rMin.y)
                {
                    camPos += new Vector3(0f, rMin.y - camMin.y, 0f);
                }

                if (camMax.y > rMax.y)
                {
                    camPos += new Vector3(0f, rMax.y - camMax.y, 0f);
                }
            }

            if (m_CameraSwizzleView != GridLayout.CellSwizzle.XYZ && m_CameraSwizzleView != GridLayout.CellSwizzle.YXZ)
            {
                if (camMin.z < rMin.z)
                {
                    camPos += new Vector3(0f, 0f, rMin.z - camMin.z);
                }
                if (camMax.z > rMax.z)
                {
                    camPos += new Vector3(0f, 0f, rMax.z - camMax.z);
                }
            }

            cam.transform.position = camPos;

            DestroyImmediate(m_GridMesh);
            m_GridMesh = null;
        }


        public Texture RenderTexture()
        {
            if (!IsGUIRectValid())
                return null;

            if (m_DelayedResetPaletteInstance)
            {
                m_DelayedResetPaletteInstance = false;
                var originalSwizzleView = m_CameraSwizzleView;
                ResetPreviewInstance();
                if (palette != null && originalSwizzleView != m_CameraSwizzleView)
                {
                    ResetPreviewCamera();
                }
            }

            if (paletteInstance == null)
                return null;

            if (m_GridMesh != null && GetGridHash() != m_LastGridHash)
            {
                ResetPreviewGridMesh();
            }

            using (new PreviewInstanceScope(guiRect, m_PreviewUtility, paletteInstance, backgroundColor, GridPaintingState.drawGizmos))
            {
                m_PreviewUtility.Render(true);
                if (GridPaintingState.drawGridGizmo)
                {
                    RenderGrid();
                }
                CallOnPaintSceneGUI(mouseGridPosition);
                if (GridPaintingState.drawGizmos)
                {
                    // Set CameraType to SceneView to force Gizmos to be drawn
                    var storedType = m_PreviewUtility.camera.cameraType;
                    m_PreviewUtility.camera.cameraType = CameraType.SceneView;
                    Handles.Internal_DoDrawGizmos(m_PreviewUtility.camera);
                    m_PreviewUtility.camera.cameraType = storedType;
                }
            }

            RenderDragAndDropPreview();
            RenderSelectedBrushMarquee();
            CallOnSceneGUI();

            var texture = m_PreviewUtility.EndPreview();
            m_LastGridHash = GetGridHash();
            return texture;
        }

        private int GetGridHash()
        {
            var gridToHash = prefabGrid;
            if (gridToHash == null)
                return 0;

            int hash = gridToHash.GetHashCode();
            unchecked
            {
                hash = hash * 33 + gridToHash.cellGap.GetHashCode();
                hash = hash * 33 + gridToHash.cellLayout.GetHashCode();
                hash = hash * 33 + gridToHash.cellSize.GetHashCode();
                hash = hash * 33 + gridToHash.cellSwizzle.GetHashCode();
                hash = hash * 33 + SceneViewGridManager.sceneViewGridComponentGizmo.Color.GetHashCode();
            }
            return hash;
        }

        private void RenderDragAndDropPreview()
        {
            if (!activeDragAndDrop || m_HoverData == null || m_HoverData.Count == 0)
                return;

            var rect = TileDragAndDrop.GetMinMaxRect(m_HoverData.Keys);
            rect.position += mouseGridPosition;
            DragAndDrop.visualMode = DragAndDropVisualMode.Copy;
            GridEditorUtility.DrawGridMarquee(grid, new BoundsInt(new Vector3Int(rect.xMin, rect.yMin, zPosition), new Vector3Int(rect.width, rect.height, 1)), Color.white);
        }

        private void RenderGrid()
        {
            // MeshTopology.Lines doesn't give nice pixel perfect grid so we have to have separate codepath with MeshTopology.Quads specially for palette window here
            if (m_GridMesh == null && grid.cellLayout == GridLayout.CellLayout.Rectangle)
            {
                m_GridMesh = GridEditorUtility.GenerateCachedGridMesh(grid, k_GridColor, 1f / LocalToScreenRatio(), paddedViewBoundsInt, grid.cellSwizzle == GridLayout.CellSwizzle.XYZ ? MeshTopology.Quads : MeshTopology.Lines);
            }
            GridEditorUtility.DrawGridGizmo(grid, grid.transform, k_GridColor, ref m_GridMesh, ref m_GridMaterial, true);
        }

        private bool IsPickingEvent<T>(T evt) where T : PointerEventBase<T>, new()
        {
            return ((evt.ctrlKey && !TilemapEditorTool.IsActive(typeof(MoveTool)))
                    || TilemapEditorTool.IsActive(typeof(PickingTool))
                    || !TilemapEditorTool.IsActive(typeof(SelectTool)) && PickingIsDefaultTool())
                   && evt.button == 0 && !evt.altKey;
        }

        private bool IsPaintingEvent<T>(T evt) where T : PointerEventBase<T>, new()
        {
            return (evt.button == 0 && !evt.ctrlKey && !evt.altKey && TilemapEditorTool.IsActive(typeof(PaintTool)));
        }

        private bool IsErasingEvent<T>(T evt) where T : PointerEventBase<T>, new()
        {
            return (evt.button == 0 && !evt.ctrlKey && !evt.altKey
                    && ((evt.shiftKey && !TilemapEditorTool.IsActive(typeof(BoxTool))
                                   && !TilemapEditorTool.IsActive(typeof(FillTool))
                                   && !TilemapEditorTool.IsActive(typeof(SelectTool))
                                   && !TilemapEditorTool.IsActive(typeof(MoveTool)))
                        || TilemapEditorTool.IsActive(typeof(EraseTool))));
        }

        private bool IsMoveEvent<T>(T evt) where T : PointerEventBase<T>, new()
        {
            return (evt.button == 0
                    && !evt.altKey
                    && TilemapEditorTool.IsActive(typeof(MoveTool)));
        }

        private bool IsSelectingEvent<T>(T evt) where T : PointerEventBase<T>, new()
        {
            return (evt.button == 0
                    && !evt.altKey
                    && (TilemapEditorTool.IsActive(typeof(SelectTool)) ||
                        (TilemapEditorTool.IsActive(typeof(MoveTool)) && evt.ctrlKey)));
        }

        private class PreviewInstanceScope : IDisposable
        {
            private readonly bool m_OldFog;
            private readonly bool m_DrawGizmos;
            private readonly Transform[] m_PaletteTransforms;

            public PreviewInstanceScope(Rect guiRect, PreviewRenderUtility previewRenderUtility, GameObject paletteInstance, Color backgroundColor, bool drawGizmos)
            {
                m_DrawGizmos = drawGizmos;
                m_OldFog = RenderSettings.fog;

                previewRenderUtility.BeginPreview(guiRect, null);

                // Draw Background here with user preference color
                Graphics.DrawTexture(new Rect(0.0f, 0.0f
                        , (float) 2 * EditorGUIUtility.pixelsPerPoint * guiRect.width
                        , (float) 2 * EditorGUIUtility.pixelsPerPoint * guiRect.height)
                    , Texture2D.grayTexture, new Rect(0.0f, 0.0f, 1f, 1f)
                    , 0, 0, 0, 0
                    , backgroundColor, null);

                Unsupported.SetRenderSettingsUseFogNoDirty(false);
                if (m_DrawGizmos)
                {
                    m_PaletteTransforms = paletteInstance.GetComponentsInChildren<Transform>();
                    foreach (var transform in m_PaletteTransforms)
                        transform.gameObject.hideFlags = HideFlags.None;
                    // Case 1199516: Set Dirty on palette instance to force a refresh on gizmo drawing
                    EditorUtility.SetDirty(paletteInstance);
                    Unsupported.SceneTrackerFlushDirty();
                }
                var renderers = paletteInstance.GetComponentsInChildren<Renderer>();
                foreach (var renderer in renderers)
                {
                    renderer.allowOcclusionWhenDynamic = false;
                }
                previewRenderUtility.AddManagedGO(paletteInstance);
                Handles.DrawCameraImpl(guiRect, previewRenderUtility.camera, DrawCameraMode.Textured, false, new DrawGridParameters(), true, false);
            }

            public void Dispose()
            {
                if (m_DrawGizmos && m_PaletteTransforms != null)
                {
                    foreach (var transform in m_PaletteTransforms)
                        transform.gameObject.hideFlags = HideFlags.HideAndDontSave;
                }
                Unsupported.SetRenderSettingsUseFogNoDirty(m_OldFog);
            }
        }

        private bool DoDragPerform()
        {
            if (palette != null && !GridPaintingState.isPaletteEditable)
                return false;

            if (m_HoverData == null)
                return false;

            if (m_HoverData.Count == 0)
            {
                FlushHoverData();
                GUI.changed = true;
                return false;
            }

            var paletteDirectory = string.Empty;
            if (palette == null)
            {
                DestroyPreviewInstance();

                var cellLayout = GridPaletteUtility.GetCellLayoutFromGridPaletteType(m_FirstUserPaletteType);
                var cellSizing = GridPalette.CellSizing.Automatic;
                var cellSize = new Vector3(1f, 1f, 0f);
                var cellSwizzle = GridLayout.CellSwizzle.XYZ;
                var sortMode = TransparencySortMode.Default;
                var sortAxis = new Vector3(0f, 0f, 1f);
                switch (m_FirstUserPaletteType)
                {
                    case GridPaletteUtility.GridPaletteType.HexagonalFlatTop:
                    {
                        cellSwizzle = GridLayout.CellSwizzle.YXZ;
                        break;
                    }
                    case GridPaletteUtility.GridPaletteType.Isometric:
                    {
                        cellSizing = GridPalette.CellSizing.Manual;
                        cellSize = new Vector3(1, 0.5f, 1);
                        break;
                    }
                    case GridPaletteUtility.GridPaletteType.IsometricZAsY:
                    {
                        cellSizing = GridPalette.CellSizing.Manual;
                        cellSize = new Vector3(1, 0.5f, 1);
                        sortMode = TransparencySortMode.CustomAxis;
                        sortAxis = new Vector3(0f, 1f, -0.25f);
                        break;
                    }
                }

                var go = GridPaletteUtility.CreateNewPaletteAtCurrentFolder("New Tile Palette"
                    , cellLayout
                    , cellSizing
                    , cellSize
                    , cellSwizzle, sortMode, sortAxis);
                var temporaryHoverData = m_HoverData;
                m_HoverData = null;
                if (go != null)
                {
                    GridPaintingState.palette = go;
                    var assetPath = AssetDatabase.GetAssetPath(go);
                    if (!String.IsNullOrEmpty(assetPath))
                    {
                        paletteDirectory = FileUtil.UnityGetDirectoryName(assetPath);
                    }
                    Selection.activeObject = go;
                }
                ResetPreviewInstance();
                m_HoverData = temporaryHoverData;
            }
            // No palette was created
            var paletteTilemap = tilemap;
            if (paletteTilemap == null)
                return false;

            RegisterUndo();

            var wasEmpty = TilemapIsEmpty(paletteTilemap);

            var targetPosition = mouseGridPosition;
            var tilemapPosition = new Vector2Int(-Int32.MaxValue, -Int32.MaxValue);
            DragAndDrop.visualMode = DragAndDropVisualMode.Copy;
            var tileSheet = TileDragAndDrop.ConvertToTileSheet(m_HoverData, paletteDirectory);
            var i = 0;
            foreach (var item in m_HoverData)
            {
                var offset = Vector3.zero;
                if (item.Value.hasOffset)
                {
                    offset = item.Value.positionOffset - paletteTilemap.tileAnchor;

                    var cellSize = paletteTilemap.cellSize;
                    if (wasEmpty)
                    {
                        cellSize = item.Value.scaleFactor;
                    }
                    offset.x *= cellSize.x;
                    offset.y *= cellSize.y;
                    offset.z *= cellSize.z;
                }

                var placePosition = targetPosition + item.Key;
                // Placing Tiles
                if (i < tileSheet.Count)
                {
                    if (tilemapPosition.y < placePosition.y ||
                        (tilemapPosition.y == placePosition.y && tilemapPosition.x > placePosition.x))
                        tilemapPosition = placePosition;

                    SetTile(paletteTilemap
                        , targetPosition + item.Key
                        , tileSheet[i++]
                        , Color.white
                        , Matrix4x4.TRS(offset, Quaternion.identity, Vector3.one));
                }
                else
                // Placing GameObjects
                {
                    if (item.Value.hoverObject is not GameObject go)
                        continue;

                    GameObject instance;
                    if (PrefabUtility.IsPartOfPrefabAsset(go))
                    {
                        instance = (GameObject) PrefabUtility.InstantiatePrefab(go, paletteTilemap.transform);
                    }
                    else
                    {
                        instance = Instantiate(go, paletteTilemap.transform);
                        instance.name = go.name;
                        instance.SetActive(true);
                        foreach (var renderer in instance.GetComponentsInChildren<Renderer>())
                        {
                            renderer.enabled = true;
                        }
                    }

                    var tileAnchor = paletteTilemap.tileAnchor;
                    var cellSize = paletteTilemap.cellSize;
                    var cellStride = cellSize + paletteTilemap.cellGap;
                    cellStride.x = Mathf.Approximately(0f, cellStride.x) ? 1f : cellStride.x;
                    cellStride.y = Mathf.Approximately(0f, cellStride.y) ? 1f : cellStride.y;
                    cellStride.z = Mathf.Approximately(0f, cellStride.z) ? 1f : cellStride.z;
                    var anchorRatio = new Vector3(
                        tileAnchor.x * cellSize.x / cellStride.x,
                        tileAnchor.y * cellSize.y / cellStride.y,
                        tileAnchor.z * cellSize.z / cellStride.z
                    );
                    instance.transform.position = grid.LocalToWorld(grid.CellToLocalInterpolated(((Vector3Int) placePosition) + anchorRatio));
                    Undo.RegisterCreatedObjectUndo(instance, "Drag and drop GameObject");
                }
            }

            if (wasEmpty && paletteTilemap.GetUsedTilesCount() != 0 &&
                GridPaintingState.gridBrush != null)
            {
                var pickBounds = new BoundsInt(new Vector3Int(tilemapPosition.x, tilemapPosition.y, 0), Vector3Int.one);
                PickBrush(pickBounds, GridPaintingState.activePivot);
            }

            OnPaletteChanged();

            m_PaletteNeedsSave = true;
            FlushHoverData();
            GUI.changed = true;
            SavePaletteIfNecessary();

            if (wasEmpty)
            {
                ResetPreviewInstance();
                FrameEntirePalette();
            }

            return true;
        }

        internal void SetTile(Tilemap tilemapTarget, Vector2Int position, TileBase tile, Color color, Matrix4x4 matrix)
        {
            Vector3Int pos3 = new Vector3Int(position.x, position.y, zPosition);
            tilemapTarget.SetTile(pos3, tile);
            tilemapTarget.SetColor(pos3, color);
            tilemapTarget.SetTransformMatrix(pos3, tilemapTarget.GetTransformMatrix(pos3) * matrix);
        }

        protected override void Paint(Vector3Int position)
        {
            if (gridBrush == null)
                return;

            gridBrush.Paint(grid, brushTarget, position);
            OnPaletteChanged();
        }

        protected override void Erase(Vector3Int position)
        {
            if (gridBrush == null)
                return;

            gridBrush.Erase(grid, brushTarget, position);
            OnPaletteChanged();
        }

        protected override void BoxFill(BoundsInt position)
        {
            if (gridBrush == null)
                return;

            gridBrush.BoxFill(grid, brushTarget, position);
            OnPaletteChanged();
        }

        protected override void BoxErase(BoundsInt position)
        {
            if (gridBrush == null)
                return;

            gridBrush.BoxErase(grid, brushTarget, position);
            OnPaletteChanged();
        }

        protected override void FloodFill(Vector3Int position)
        {
            if (gridBrush == null)
                return;

            gridBrush.FloodFill(grid, brushTarget, position);
            OnPaletteChanged();
        }

        protected override void PickBrush(BoundsInt position, Vector3Int pickingStart)
        {
            if (grid == null || gridBrush == null)
                return;

            gridBrush.Pick(grid, brushTarget, position, pickingStart);

            if (!InGridEditMode())
                TilemapEditorTool.SetActiveEditorTool(typeof(PaintTool));

            GridPaintingState.hasActivePick = true;
            GridPaintingState.activePick = position;
            GridPaintingState.activePivot = pickingStart;
            GridPaintingState.activePickPalette = GridPaintingState.palette;

            if (!m_DisableOnBrushPicked)
                onBrushPicked?.Invoke();
        }

        protected override void Select(BoundsInt position)
        {
            if (grid)
            {
                GridSelection.Select(brushTarget, position);
                gridBrush.Select(grid, brushTarget, position);
            }
        }

        protected override void Move(BoundsInt from, BoundsInt to)
        {
            if (grid)
                gridBrush.Move(grid, brushTarget, from, to);
        }

        protected override void MoveStart(BoundsInt position)
        {
            if (grid)
                gridBrush.MoveStart(grid, brushTarget, position);
        }

        protected override void MoveEnd(BoundsInt position)
        {
            if (grid)
            {
                gridBrush.MoveEnd(grid, brushTarget, position);
                OnPaletteChanged();
            }
        }

        protected override bool CustomTool(bool isToolHotControl, TilemapEditorTool tool, Vector3Int position)
        {
            var executed = false;
            if (grid)
            {
                executed = tool.HandleTool(isToolHotControl, grid, brushTarget, position);
                if (executed)
                    OnPaletteChanged();
            }
            return executed;
        }

        protected bool IsMouseUpInWindow()
        {
            return guiRect.Contains(mousePosition);
        }

        public override void Repaint()
        {
            attachedVisualElement?.MarkDirtyRepaint();
        }

        protected override void ClearGridSelection()
        {
            GridSelection.Clear();
        }

        public override bool isActive => grid != null;

        public override Rect rectPosition => attachedVisualElement.worldBound;

        public override VisualElement windowRoot => attachedVisualElement?.GetRoot();

        protected override void OnBrushPickStarted()
        {
        }

        protected override void OnBrushPickDragged(BoundsInt position)
        {
            GridPaintingState.hasActivePick = true;
            GridPaintingState.activePick = position;
            GridPaintingState.activePickPalette = GridPaintingState.palette;
        }

        protected override void OnBrushPickCancelled()
        {
            GridPaintingState.hasActivePick = false;
            GridPaintingState.activePick = new BoundsInt(Vector3Int.zero, Vector3Int.zero);
            GridPaintingState.activePivot = Vector3Int.zero;
            GridPaintingState.activePickPalette = null;
        }

        private void PingTileAsset(RectInt rect)
        {
            // Only able to ping asset if only one asset is selected
            if (rect.size == Vector2Int.zero && tilemap != null)
            {
                TileBase tile = tilemap.GetTile(new Vector3Int(rect.xMin, rect.yMin, zPosition));
                EditorGUIUtility.PingObject(tile);
                Selection.activeObject = tile;
            }
        }

        protected override bool ValidateFloodFillPosition(Vector3Int position)
        {
            return true;
        }

        protected override bool PickingIsDefaultTool()
        {
            return !m_Unlocked;
        }

        protected override bool CanPickOutsideEditMode()
        {
            return true;
        }

        protected override GridLayout.CellLayout CellLayout()
        {
            if (grid != null)
                return grid.cellLayout;
            return GridLayout.CellLayout.Rectangle;
        }

        protected override Vector2Int ScreenToGrid(Vector2 screenPosition, float zPosition)
        {
            Vector3 local = ScreenToLocal(screenPosition);
            if (GridPaintingState.gridBrushMousePositionAtZ)
                local.z = zPosition;
            var localS = Grid.Swizzle(m_CameraSwizzleView, local);
            var result3 = grid.LocalToCell(localS);
            var result = new Vector2Int(result3.x, result3.y);
            return result;
        }

        private void RenderSelectedBrushMarquee()
        {
            if (!activeDragAndDrop && !unlocked && GridPaintingState.hasActivePick)
            {
                DrawSelectionGizmo(GridPaintingState.activePick);
            }
        }

        private void DrawSelectionGizmo(BoundsInt selectionBounds)
        {
            if (!GUI.enabled)
                return;

            var color = Color.white;
            if (isPicking)
                color = Color.cyan;

            GridEditorUtility.DrawGridMarquee(grid, new BoundsInt(new Vector3Int(selectionBounds.xMin, selectionBounds.yMin, 0), new Vector3Int(selectionBounds.size.x, selectionBounds.size.y, 1)), color);
        }

        private void CallOnSceneGUI()
        {
            var gridLayout = tilemap != null ? tilemap : grid as GridLayout;
            var hasSelection = GridSelection.active  && GridSelection.target == brushTarget;
            if (hasSelection)
            {
                var rect = new RectInt(GridSelection.position.xMin, GridSelection.position.yMin, GridSelection.position.size.x, GridSelection.position.size.y);
                var brushBounds = new BoundsInt(new Vector3Int(rect.x, rect.y, zPosition), new Vector3Int(rect.width, rect.height, 1));
                GridBrushEditorBase.OnSceneGUIInternal(gridLayout, brushTarget, brushBounds, EditTypeToBrushTool(ToolManager.activeToolType), m_MarqueeStart.HasValue || executing);
            }
            if (GridPaintingState.activeBrushEditor != null)
            {
                GridPaintingState.activeBrushEditor.OnSceneGUI(gridLayout, brushTarget);
                if (hasSelection)
                {
                    GridPaintingState.activeBrushEditor.OnSelectionSceneGUI(gridLayout, brushTarget);
                    if (GridSelectionTool.IsActive() && unlocked)
                    {
                        var tool = EditorToolManager.activeTool as GridSelectionTool;
                        tool.OnToolGUI();
                    }
                }
            }
        }

        private void CallOnPaintSceneGUI(Vector2Int position)
        {
            if (activeDragAndDrop)
                return;

            if (!unlocked && !TilemapEditorTool.IsActive(typeof(SelectTool)) && !TilemapEditorTool.IsActive(typeof(PickingTool)))
                return;

            var hasSelection = GridSelection.active && GridSelection.target == brushTarget;
            if ((!hasSelection || !isPicking) && GridPaintingState.activeGrid != this)
                return;

            var brush = GridPaintingState.gridBrush;
            if (brush == null)
                return;

            var rect = new RectInt(position, new Vector2Int(1, 1));

            if (m_MarqueeStart.HasValue)
                rect = GridEditorUtility.GetMarqueeRect(position, m_MarqueeStart.Value);
            else if (hasSelection)
                rect = new RectInt(GridSelection.position.xMin, GridSelection.position.yMin, GridSelection.position.size.x, GridSelection.position.size.y);

            var gridLayout = tilemap != null ? tilemap.layoutGrid : grid as GridLayout;
            var brushBounds = new BoundsInt(new Vector3Int(rect.x, rect.y, zPosition), new Vector3Int(rect.width, rect.height, 1));

            if (GridPaintingState.activeBrushEditor != null)
                GridPaintingState.activeBrushEditor.OnPaintSceneGUI(gridLayout, brushTarget, brushBounds,
                    EditTypeToBrushTool(ToolManager.activeToolType),
                    m_MarqueeStart.HasValue || executing);
            else // Fallback when user hasn't defined custom editor
                GridBrushEditorBase.OnPaintSceneGUIInternal(gridLayout, Selection.activeGameObject, brushBounds,
                    EditTypeToBrushTool(ToolManager.activeToolType),
                    m_MarqueeStart.HasValue || executing);
        }

        protected override void RegisterUndo()
        {
            if (palette != null && paletteInstance != null)
                Undo.RegisterFullObjectHierarchyUndo(paletteInstance, "Edit Palette");
        }

        private void OnPaletteChanged()
        {
            m_PaletteUsed = true;
            m_PaletteNeedsSave = true;
            Undo.FlushUndoRecordObjects();
        }

        public void CheckRevertIfChanged(string[] paths)
        {
            if (paths != null && m_PaletteNeedsSave && palette != null)
            {
                var currentPalettePath = AssetDatabase.GetAssetPath(palette);
                foreach (var path in paths)
                {
                    if (currentPalettePath == path)
                    {
                        m_PaletteNeedsSave = false;
                        ResetPreviewInstance();
                        Debug.LogWarningFormat(palette, paletteSavedOutsideClipboard, palette.name);
                        break;
                    }
                }
            }
        }

        public bool SavePaletteIfNecessary()
        {
            bool needsSave = m_PaletteNeedsSave;
            if (needsSave)
            {
                SavePalette();
                m_PaletteNeedsSave = false;
            }
            return needsSave;
        }

        private void SavePalette()
        {
            if (palette != null && paletteInstance != null && GridPaintingState.isPaletteEditable)
            {
                TilePaletteSaveUtility.SaveTilePalette(palette, paletteInstance);
                ResetPreviewInstance();
                Repaint();
            }
        }

        public Vector2 GridToScreen(Vector2 gridPosition)
        {
            var gridPosition3 = new Vector3(gridPosition.x, gridPosition.y, 0);
            return LocalToScreen(grid.CellToLocalInterpolated(gridPosition3));
        }

        public Vector2 ScreenToLocal(Vector2 screenPosition)
        {
            var viewPosition = m_PreviewUtility.camera.transform.position;
            Vector2 viewXYPosition = Grid.InverseSwizzle(m_CameraSwizzleView, viewPosition);
            screenPosition -= new Vector2(guiRect.xMin, guiRect.yMin);
            var offsetFromCenter = new Vector2(screenPosition.x - guiRect.width * .5f, guiRect.height * .5f - screenPosition.y);
            return viewXYPosition + offsetFromCenter / LocalToScreenRatio();
        }

        protected Vector2 LocalToScreen(Vector2 localPosition)
        {
            var viewPosition = m_PreviewUtility.camera.transform.position;
            Vector2 viewXYPosition = Grid.InverseSwizzle(m_CameraSwizzleView, viewPosition);
            var offsetFromCenter = new Vector2(localPosition.x - viewXYPosition.x, viewXYPosition.y - localPosition.y);
            return offsetFromCenter * LocalToScreenRatio() + new Vector2(guiRect.width * .5f + guiRect.xMin, guiRect.height * .5f + guiRect.yMin);
        }

        private float LocalToScreenRatio()
        {
            return guiRect.height / (m_PreviewUtility.camera.orthographicSize * 2f);
        }

        private float LocalToScreenRatio(float viewHeight)
        {
            return viewHeight / (m_PreviewUtility.camera.orthographicSize * 2f);
        }

        private static bool TilemapIsEmpty(Tilemap tilemap)
        {
            return tilemap.GetUsedTilesCount() == 0;
        }

        public void UnlockAndEdit()
        {
            unlocked = true;
            m_PaletteNeedsSave = true;
        }

        internal void PickFirstFromPalette()
        {
            if (tilemap == null)
                return;

            var pickBounds = tilemap.cellBounds;
            pickBounds.size = Vector3Int.one;
            PickBrush(pickBounds, Vector3Int.zero);

            if (!TilemapEditorTool.IsActive(typeof(PaintTool)))
                TilemapEditorTool.SetActiveEditorTool(typeof(PaintTool));
        }

        // TODO: Better way of clearing caches than AssetPostprocessor
        public class AssetProcessor : AssetPostprocessor
        {
            public override int GetPostprocessOrder()
            {
                return int.MaxValue;
            }

            private static void OnPostprocessAllAssets(string[] importedAssets, string[] deletedAssets, string[] movedAssets, string[] movedFromPath)
            {
                foreach (var clipboard in instances)
                {
                    clipboard.DelayedResetPreviewInstance();
                }
            }
        }

        public class PaletteAssetModificationProcessor : AssetModificationProcessor
        {
            static void OnWillCreateAsset(string assetName)
            {
                SavePalettesIfRequired(null);
            }

            static string[] OnWillSaveAssets(string[] paths)
            {
                SavePalettesIfRequired(paths);
                return paths;
            }

            static void SavePalettesIfRequired(string[] paths)
            {
                if (GridPaintingState.savingPalette)
                    return;

                foreach (var clipboard in instances)
                {
                    if (clipboard.isModified)
                    {
                        clipboard.CheckRevertIfChanged(paths);
                        clipboard.SavePaletteIfNecessary();
                        clipboard.Repaint();
                    }
                }
            }
        }

    }
}
