using System;
using System.Linq;
using UnityEditor.Toolbars;
using UnityEngine;
using UnityEngine.UIElements;

namespace UnityEditor.Tilemaps
{
    /// <summary>
    /// Popup Field for selecting the Active Palette for Grid Painting.
    /// </summary>
    [EditorToolbarElement(k_ToolbarId)]
    [UxmlElement]
    public sealed partial class TilePaletteActivePalettePopup : PopupField<GameObject>
    {
        internal const string k_ToolbarId = "Tools/Tile Palette Active Palette";

        private static string kNullGameObjectName = L10n.Tr("Create New Tile Palette");
        private static string kLabelTooltip =
            L10n.Tr("Specifies the currently active Palette used for painting in the Scene View.");

        /// <summary>
        /// Factory for TilePaletteActivePalettePopup.
        /// </summary>
        [Obsolete("TilePaletteActivePalettePopupFactory is deprecated and will be removed. Use UxmlElementAttribute instead.", false)]
        public class TilePaletteActivePalettePopupFactory : UxmlFactory<TilePaletteActivePalettePopup, TilePaletteActivePalettePopupUxmlTraits> {}
        /// <summary>
        /// UxmlTraits for TilePaletteActivePalettePopup.
        /// </summary>
        [Obsolete("TilePaletteActivePalettePopupUxmlTraits is deprecated and will be removed. Use UxmlElementAttribute instead.", false)]
        public class TilePaletteActivePalettePopupUxmlTraits : UxmlTraits {}

        /// <summary>
        /// USS class name of elements of this type.
        /// </summary>
        private new static readonly string ussClassName = "unity-tilepalette-activepalettes-field";
        /// <summary>
        /// USS class name of labels in elements of this type.
        /// </summary>
        private new static readonly string labelUssClassName = ussClassName + "__label";
        /// <summary>
        /// USS class name of input elements in elements of this type.
        /// </summary>
        private new static readonly string inputUssClassName = ussClassName + "__input";

        /// <summary>
        /// Initializes and returns an instance of TilePaletteActivePalettesPopup.
        /// </summary>
        public TilePaletteActivePalettePopup() : this(null) {}

        /// <summary>
        /// Initializes and returns an instance of TilePaletteActivePalettesPopup.
        /// </summary>
        /// <param name="label">Label name for the Popup</param>
        public TilePaletteActivePalettePopup(string label)
            : base(label, GridPaintingState.palettes.ToList(), GetActivePaletteIndex())
        {
            AddToClassList(ussClassName);
            labelElement.AddToClassList(labelUssClassName);
            visualInput.AddToClassList(inputUssClassName);

            TilePaletteOverlayUtility.SetStyleSheet(this);

            labelElement.tooltip = kLabelTooltip;

            RegisterCallback<AttachToPanelEvent>(OnAttachedToPanel);
            RegisterCallback<DetachFromPanelEvent>(OnDetachFromPanel);

            m_FormatSelectedValueCallback += FormatSelectedValueCallback;
            createMenuCallback += CreateMenuCallback;

            UpdateTargets();
            SetValueWithoutNotify(GridPaintingState.palette);
        }

        private void OnAttachedToPanel(AttachToPanelEvent evt)
        {
            GridPaintingState.paletteChanged += OnPaletteChanged;
            GridPaintingState.palettesChanged += UpdateTargets;
        }

        private void OnDetachFromPanel(DetachFromPanelEvent evt)
        {
            GridPaintingState.palettesChanged += UpdateTargets;
            GridPaintingState.paletteChanged -= OnPaletteChanged;
        }

        private void OnPaletteChanged(GameObject _)
        {
            UpdateActivePalette();
        }

        private string FormatSelectedValueCallback(GameObject go)
        {
            if (go != null)
                return go.name;
            return kNullGameObjectName;
        }

        private IGenericMenu CreateMenuCallback()
        {
            return new TilePaletteActivePaletteDropdownMenu();
        }

        private static int GetActivePaletteIndex()
        {
            return GridPaintingState.palettes.IndexOf(GridPaintingState.palette);
        }

        private void UpdateChoices()
        {
            choices.Clear();
            foreach (var target in GridPaintingState.palettes)
            {
                choices.Add(target);
            }
            formatSelectedValueCallback = FormatSelectedValueCallback;
        }

        private void UpdateActivePalette()
        {
            var newIndex = GetActivePaletteIndex();
            if (newIndex != -1 && choices.Count < newIndex)
            {
                UpdateChoices();
                newIndex = GetActivePaletteIndex();
            }
            index = newIndex;
        }

        private void UpdateTargets()
        {
            UpdateChoices();
            UpdateActivePalette();
        }
    }

    [EditorToolbarElement(k_ToolbarId)]
    internal class TilePaletteActivePalettePopupIcon : VisualElement
    {
        internal const string k_ToolbarId = "Tools/Tile Palette Active Palette Icon";

        private static string kTooltip = L10n.Tr("Active Palette");

        /// <summary>
        /// USS class name of elements of this type.
        /// </summary>
        public static readonly string ussClassName = "unity-tilepalette-activepalette-icon";

        private readonly string k_IconPath = "Packages/com.unity.2d.tilemap/Editor/Icons/Tilemap.TilePalette.png";

        public TilePaletteActivePalettePopupIcon()
        {
            AddToClassList(ussClassName);
            TilePaletteOverlayUtility.SetStyleSheet(this);

            style.backgroundImage = EditorGUIUtility.LoadIcon(k_IconPath);
            tooltip = kTooltip;
        }
    }
}
