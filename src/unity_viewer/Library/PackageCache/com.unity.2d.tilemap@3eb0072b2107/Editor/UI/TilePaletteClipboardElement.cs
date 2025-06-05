using System;
using UnityEngine;
using UnityEngine.UIElements;
using UnityEditor.EditorTools;

namespace UnityEditor.Tilemaps
{
    /// <summary>
    /// A Visual Element which handles and displays a Tile Palette Clipboard.
    /// A Tile Palette Clipboard shows the Active Palette for Grid Painting and allows
    /// users to use the Active Brush to assign and pick items for painting.
    /// </summary>
    [UxmlElement]
    public partial class TilePaletteClipboardElement : VisualElement
    {
        /// <summary>
        /// Factory for TilePaletteClipboardElement.
        /// </summary>
        [Obsolete("TilePaletteClipboardElementFactory is deprecated and will be removed. Use UxmlElementAttribute instead.", false)]
        public class TilePaletteClipboardElementFactory : UxmlFactory<TilePaletteClipboardElement, TilePaletteClipboardElementUxmlTraits> {}
        /// <summary>
        /// UxmlTraits for TilePaletteClipboardElement.
        /// </summary>
        [Obsolete("TilePaletteClipboardElementUxmlTraits is deprecated and will be removed. Use UxmlElementAttribute instead.", false)]
        public class TilePaletteClipboardElementUxmlTraits : UxmlTraits {}

        internal static readonly string overlayClassName = "unity-tilepalette-clipboard-element-overlay";

        private static readonly string ussClassName = "unity-tilepalette-clipboard-element";
        private static readonly string k_Name = L10n.Tr("Tile Palette Clipboard Element");

        private GridPaintPaletteClipboard m_TilePaletteClipboard;
        private EditorWindow m_Window;

        /// <summary>
        /// Callback when the active Brush does a Pick on the Clipboard.
        /// </summary>
        public event Action onBrushPicked;

        /// <summary>
        /// Whether the clipboard is unlocked for editing.
        /// </summary>
        public bool clipboardUnlocked
        {
            get => m_TilePaletteClipboard.unlocked;
            set => m_TilePaletteClipboard.unlocked = value;
        }

        /// <summary>
        /// The last active grid position on the clipboard.
        /// </summary>
        public Vector3Int clipboardMouseGridPosition => new Vector3Int(m_TilePaletteClipboard.mouseGridPosition.x, m_TilePaletteClipboard.mouseGridPosition.y, m_TilePaletteClipboard.zPosition);

        /// <summary>
        /// Callback when the clipboard unlock status has changed
        /// </summary>
        public event Action<bool> clipboardUnlockedChanged;

        private bool clipboardIsNotActive => m_TilePaletteClipboard == null || !m_ClipboardImageElement.visible;

        internal GridPaintPaletteClipboard clipboardView => m_TilePaletteClipboard;

        private TilePaletteClipboardFirstUserElement m_FirstUserElement;
        private TilePaletteClipboardErrorElement m_ErrorElement;
        private IMGUIContainer m_GizmoHandlerElement;
        private Image m_ClipboardImageElement;

        /// <summary>
        /// Initializes and returns an instance of TilePaletteClipboardElement.
        /// </summary>
        public TilePaletteClipboardElement()
        {
            AddToClassList(ussClassName);

            name = k_Name;
            TilePaletteOverlayUtility.SetStyleSheet(this);

            RegisterCallback<AttachToPanelEvent>(OnAttachedToPanel);
            RegisterCallback<DetachFromPanelEvent>(OnDetachFromPanel);

            m_GizmoHandlerElement = new IMGUIContainer(HandleIMGUI);
            m_GizmoHandlerElement.style.flexGrow = 1.0f;

            m_ClipboardImageElement = new Image();
            m_ClipboardImageElement.style.backgroundColor = GridPaintPaletteClipboard.tilePaletteBackgroundColor.Color;
            m_ClipboardImageElement.style.flexGrow = 1.0f;
            m_ClipboardImageElement.focusable = true;

            m_ClipboardImageElement.Add(m_GizmoHandlerElement);
            Add(m_ClipboardImageElement);

            m_ErrorElement = new TilePaletteClipboardErrorElement();
            m_ErrorElement.style.display = DisplayStyle.None;
            m_ErrorElement.style.visibility = Visibility.Hidden;
            SetEmptyText();
            Add(m_ErrorElement);

            m_FirstUserElement = new TilePaletteClipboardFirstUserElement();
            m_FirstUserElement.style.display = DisplayStyle.None;
            m_FirstUserElement.style.visibility = Visibility.Hidden;
            Add(m_FirstUserElement);
        }

        private void UnlockChanged(bool unlocked)
        {
            clipboardUnlockedChanged?.Invoke(unlocked);
            CheckPaletteState();
        }

        private void OnAttachedToPanel(AttachToPanelEvent evt)
        {
            if (EditorApplication.isPlayingOrWillChangePlaymode && !EditorApplication.isPlaying)
            {
                // Delay AttachToPanel if Editor is entering playmode
                EditorApplication.delayCall += AttachToPanel;
            }
            else
            {
                AttachToPanel();
            }
        }

        private void AttachToPanel()
        {
            if (m_TilePaletteClipboard == null)
            {
                m_TilePaletteClipboard = ScriptableObject.CreateInstance<GridPaintPaletteClipboard>();
                m_TilePaletteClipboard.hideFlags = HideFlags.HideAndDontSave;
                m_TilePaletteClipboard.unlockedChanged += UnlockChanged;
                m_TilePaletteClipboard.unlocked = false;
                m_TilePaletteClipboard.attachedVisualElement = this;

                var guiRect = new Rect(0, 0, layout.width, layout.height);
                m_TilePaletteClipboard.guiRect = guiRect;

                CheckPaletteState(m_TilePaletteClipboard.paletteInstance);
            }

            RegisterCallback<GeometryChangedEvent>(OnGeometryChangedEvent);
            RegisterCallback<ValidateCommandEvent>(OnValidateCommandEvent);
            RegisterCallback<ExecuteCommandEvent>(OnExecuteCommandEvent);

            RegisterCallback<GeometryChangedEvent>(OnGeometryChangedEvent);
            RegisterCallback<WheelEvent>(OnWheelEvent);
            RegisterCallback<PointerDownEvent>(OnPointerDownEvent);
            RegisterCallback<PointerMoveEvent>(OnPointerMoveEvent);
            RegisterCallback<PointerUpEvent>(OnPointerUpEvent);
            RegisterCallback<PointerEnterEvent>(OnPointerEnterEvent);
            RegisterCallback<PointerLeaveEvent>(OnPointerLeaveEvent);
            RegisterCallback<KeyDownEvent>(OnKeyDownEvent);
            RegisterCallback<KeyUpEvent>(OnKeyUpEvent);
            RegisterCallback<DragEnterEvent>(OnDragEnterEvent);
            RegisterCallback<DragUpdatedEvent>(OnDragUpdatedEvent);
            RegisterCallback<DragPerformEvent>(OnDragPerformEvent);
            RegisterCallback<DragLeaveEvent>(OnDragLeaveEvent);
            RegisterCallback<DragExitedEvent>(OnDragExitedEvent);

            m_FirstUserElement.firstUserPaletteType = m_TilePaletteClipboard.firstUserPaletteType;
            m_FirstUserElement.onFirstUserPaletteTypeChanged += OnFirstUserPaletteTypeChanged;

            ToolManager.activeToolChanged += ActiveToolChanged;
            GridPaintingState.beforePaletteChanged += BeforePaletteChanged;
            GridPaintingState.paletteChanged += PaletteChanged;
            GridPaintingState.palettesChanged += CheckPaletteState;
        }

        private void OnDetachFromPanel(DetachFromPanelEvent evt)
        {
            UnregisterCallback<GeometryChangedEvent>(OnGeometryChangedEvent);
            UnregisterCallback<ValidateCommandEvent>(OnValidateCommandEvent);
            UnregisterCallback<ExecuteCommandEvent>(OnExecuteCommandEvent);

            UnregisterCallback<GeometryChangedEvent>(OnGeometryChangedEvent);
            UnregisterCallback<WheelEvent>(OnWheelEvent);
            UnregisterCallback<PointerDownEvent>(OnPointerDownEvent);
            UnregisterCallback<PointerMoveEvent>(OnPointerMoveEvent);
            UnregisterCallback<PointerUpEvent>(OnPointerUpEvent);
            UnregisterCallback<PointerEnterEvent>(OnPointerEnterEvent);
            UnregisterCallback<PointerLeaveEvent>(OnPointerLeaveEvent);
            UnregisterCallback<KeyDownEvent>(OnKeyDownEvent);
            UnregisterCallback<KeyUpEvent>(OnKeyUpEvent);
            UnregisterCallback<DragEnterEvent>(OnDragEnterEvent);
            UnregisterCallback<DragUpdatedEvent>(OnDragUpdatedEvent);
            UnregisterCallback<DragPerformEvent>(OnDragPerformEvent);
            UnregisterCallback<DragExitedEvent>(OnDragExitedEvent);
            UnregisterCallback<DragLeaveEvent>(OnDragLeaveEvent);

            m_FirstUserElement.onFirstUserPaletteTypeChanged -= OnFirstUserPaletteTypeChanged;

            if (m_TilePaletteClipboard != null)
                m_TilePaletteClipboard.unlockedChanged -= UnlockChanged;

            ToolManager.activeToolChanged -= ActiveToolChanged;
            GridPaintingState.beforePaletteChanged -= BeforePaletteChanged;
            GridPaintingState.paletteChanged -= PaletteChanged;
            GridPaintingState.palettesChanged -= CheckPaletteState;

            Cleanup();
        }

        private void OnFirstUserPaletteTypeChanged(GridPaletteUtility.GridPaletteType paletteType)
        {
            if (m_TilePaletteClipboard != null)
                m_TilePaletteClipboard.firstUserPaletteType = paletteType;
        }

        private void OnGeometryChangedEvent(GeometryChangedEvent evt)
        {
            if (m_TilePaletteClipboard == null)
                return;

            var guiRect = new Rect(0, 0, layout.width, layout.height);
            m_TilePaletteClipboard.guiRect = guiRect;
        }

        private void ActiveToolChanged()
        {
            CheckPaletteState();
        }

        private void HandleIMGUI()
        {
            if (clipboardIsNotActive)
                return;

            var texture = m_TilePaletteClipboard.HandleIMGUI();
            if (texture != null)
            {
                m_ClipboardImageElement.style.backgroundColor = m_TilePaletteClipboard.backgroundColor;
                m_ClipboardImageElement.image = texture;
            }
        }

        private void HandleRepaint()
        {
            MarkDirtyRepaint();
            m_ClipboardImageElement.MarkDirtyRepaint();
        }

        private void OnExecuteCommandEvent(ExecuteCommandEvent evt)
        {
            if (clipboardIsNotActive)
                return;

            m_TilePaletteClipboard.HandleExecuteCommandEvent(evt);
        }

        private void OnValidateCommandEvent(ValidateCommandEvent evt)
        {
            if (clipboardIsNotActive)
                return;

            m_TilePaletteClipboard.HandleValidateCommandEvent(evt);
        }

        private void OnWheelEvent(WheelEvent evt)
        {
            if (clipboardIsNotActive)
                return;

            m_TilePaletteClipboard.HandleWheelEvent(evt.delta, evt.mousePosition, evt.shiftKey);
            evt.StopPropagation();
            HandleRepaint();
        }

        private void OnPointerDownEvent(PointerDownEvent evt)
        {
            if (clipboardIsNotActive)
                return;

            m_TilePaletteClipboard.HandlePointerDownEvent(evt
                , evt.button
                , evt.altKey
                , evt.ctrlKey
                , evt.localPosition);
            HandleRepaint();
        }

        private void OnPointerMoveEvent(PointerMoveEvent evt)
        {
            if (clipboardIsNotActive)
                return;

            m_TilePaletteClipboard.HandlePointerMoveEvent(evt
                , evt.button
                , evt.altKey
                , evt.localPosition
                , evt.deltaPosition);
            HandleRepaint();
        }

        private void OnPointerUpEvent(PointerUpEvent evt)
        {
            if (clipboardIsNotActive)
                return;

            if (onBrushPicked != null && m_TilePaletteClipboard != null)
                m_TilePaletteClipboard.onBrushPicked += onBrushPicked;
            m_TilePaletteClipboard.HandlePointerUpEvent(evt);
            if (onBrushPicked != null && m_TilePaletteClipboard != null)
                m_TilePaletteClipboard.onBrushPicked -= onBrushPicked;
            HandleRepaint();
        }

        private void OnPointerEnterEvent(PointerEnterEvent evt)
        {
            if (m_TilePaletteClipboard == null)
                return;

            if (ClassListContains(overlayClassName))
            {
                m_ClipboardImageElement.Focus();
            }
            m_TilePaletteClipboard.HandlePointerEnterEvent(evt);
            HandleRepaint();
        }

        private void OnPointerLeaveEvent(PointerLeaveEvent evt)
        {
            if (m_TilePaletteClipboard == null)
                return;

            m_TilePaletteClipboard.HandlePointerLeaveEvent(evt);
            HandleRepaint();
        }

        private void OnKeyDownEvent(KeyDownEvent evt)
        {
            if (clipboardIsNotActive)
                return;

            m_TilePaletteClipboard.HandleKeyDownEvent(evt);
            HandleRepaint();
        }

        private void OnKeyUpEvent(KeyUpEvent evt)
        {
            if (clipboardIsNotActive)
                return;

            m_TilePaletteClipboard.HandleKeyUpEvent();
            HandleRepaint();
        }

        private void OnDragEnterEvent(DragEnterEvent evt)
        {
            if (m_TilePaletteClipboard == null)
                return;

            m_TilePaletteClipboard.HandleDragEnterEvent(evt);
            CheckPaletteState(m_TilePaletteClipboard.paletteInstance);
            HandleRepaint();
        }

        private void OnDragUpdatedEvent(DragUpdatedEvent evt)
        {
            if (clipboardIsNotActive)
                return;

            m_TilePaletteClipboard.HandleDragUpdatedEvent(evt);
            HandleRepaint();
        }

        private void OnDragPerformEvent(DragPerformEvent evt)
        {
            if (clipboardIsNotActive)
                return;

            m_TilePaletteClipboard.HandleDragPerformEvent(evt);
            CheckPaletteState(m_TilePaletteClipboard.paletteInstance);
            HandleRepaint();
        }

        private void OnDragLeaveEvent(DragLeaveEvent evt)
        {
            if (clipboardIsNotActive)
                return;

            m_TilePaletteClipboard.HandleDragLeaveEvent(evt);
            CheckPaletteState(m_TilePaletteClipboard.paletteInstance);
            HandleRepaint();
        }

        private void OnDragExitedEvent(DragExitedEvent evt)
        {
            if (clipboardIsNotActive)
                return;

            m_TilePaletteClipboard.HandleDragExitedEvent(evt);
            HandleRepaint();
        }

        /// <summary>
        /// Handles cleanup for the Tile Palette Clipboard.
        /// </summary>
        private void Cleanup()
        {
            UnityEngine.Object.DestroyImmediate(m_TilePaletteClipboard);
            m_TilePaletteClipboard = null;
        }

        private void BeforePaletteChanged()
        {
            if (m_TilePaletteClipboard == null)
                return;
            m_TilePaletteClipboard.OnBeforePaletteSelectionChanged();
        }

        private void PaletteChanged(GameObject palette)
        {
            if (m_TilePaletteClipboard == null)
                return;
            m_TilePaletteClipboard.OnAfterPaletteSelectionChanged();
            CheckPaletteState(palette);
        }

        internal void CheckPaletteState()
        {
            CheckPaletteState(GridPaintingState.palette);
        }

        private void CheckPaletteState(GameObject palette)
        {
            if (palette == null && GridPaintingState.palettes.Count == 0)
            {
                m_ClipboardImageElement.style.display = DisplayStyle.None;
                m_ClipboardImageElement.style.visibility = Visibility.Hidden;
                m_ErrorElement.style.display = DisplayStyle.None;
                m_ErrorElement.style.visibility = Visibility.Hidden;
                m_ErrorElement.ClearText();
                m_FirstUserElement.style.display = DisplayStyle.Flex;
                m_FirstUserElement.style.visibility = Visibility.Visible;
            }
            else if (palette == null && GridPaintingState.palettes.Count > 0)
            {
                m_ClipboardImageElement.style.display = DisplayStyle.None;
                m_ClipboardImageElement.style.visibility = Visibility.Hidden;
                m_FirstUserElement.style.display = DisplayStyle.None;
                m_FirstUserElement.style.visibility = Visibility.Hidden;
                m_ErrorElement.style.display = DisplayStyle.Flex;
                m_ErrorElement.style.visibility = Visibility.Visible;
                m_ErrorElement.SetInvalidPaletteText();
            }
            else if (m_TilePaletteClipboard.activeDragAndDrop && m_TilePaletteClipboard.invalidDragAndDrop)
            {
                m_ClipboardImageElement.style.display = DisplayStyle.None;
                m_ClipboardImageElement.style.visibility = Visibility.Hidden;
                m_FirstUserElement.style.display = DisplayStyle.None;
                m_FirstUserElement.style.visibility = Visibility.Hidden;
                m_ErrorElement.style.display = DisplayStyle.Flex;
                m_ErrorElement.style.visibility = Visibility.Visible;
                m_ErrorElement.SetInvalidDragAndDropText();
            }
            else if (palette.GetComponent<Grid>() == null)
            {
                m_ClipboardImageElement.style.display = DisplayStyle.None;
                m_ClipboardImageElement.style.visibility = Visibility.Hidden;
                m_FirstUserElement.style.display = DisplayStyle.None;
                m_FirstUserElement.style.visibility = Visibility.Hidden;
                m_ErrorElement.style.display = DisplayStyle.Flex;
                m_ErrorElement.style.visibility = Visibility.Visible;
                m_ErrorElement.SetInvalidGridText();
            }
            else if (m_TilePaletteClipboard.ShowNewEmptyClipboardInfo(palette))
            {
                m_ClipboardImageElement.style.display = DisplayStyle.None;
                m_ClipboardImageElement.style.visibility = Visibility.Hidden;
                m_FirstUserElement.style.display = DisplayStyle.None;
                m_FirstUserElement.style.visibility = Visibility.Hidden;
                m_ErrorElement.style.display = DisplayStyle.Flex;
                m_ErrorElement.style.visibility = Visibility.Visible;
                SetEmptyText();
            }
            else
            {
                m_ErrorElement.style.display = DisplayStyle.None;
                m_ErrorElement.style.visibility = Visibility.Hidden;
                m_ErrorElement.ClearText();
                m_FirstUserElement.style.display = DisplayStyle.None;
                m_FirstUserElement.style.visibility = Visibility.Hidden;
                m_ClipboardImageElement.style.display = DisplayStyle.Flex;
                m_ClipboardImageElement.style.visibility = Visibility.Visible;
            }
            HandleRepaint();
        }

        private void SetEmptyText()
        {
            if (GridPaintingState.isPaletteEditable)
            {
                m_ErrorElement.SetEmptyPaletteText();
            }
            else
            {
                m_ErrorElement.SetEmptyModelPaletteText();
            }
        }
    }
}
