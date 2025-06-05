using System;
using UnityEngine;
using UnityEditor.Toolbars;
using UnityEngine.UIElements;

namespace UnityEditor.Tilemaps
{
    internal class GridPaintPaletteWindowActiveTargets : VisualElement
    {
        const string kUssClassName = "unity-tilepalette-activetargets";
        const string kPopupUssClassName = "unity-tilepalette-activetargets-popup";
        const string kInfoUssClassName = "unity-tilepalette-activetargets-info";
        const string kCreateHintUssClassName = "unity-tilepalette-activetargets-info__create";

        private static string[] k_RightToolbarElements = new[] {
            TilePaletteClipboardToggle.k_ToolbarId
            , TilePaletteBrushPickToggle.k_ToolbarId
        };

        private static string k_CreateTargetTooltip =
            L10n.Tr("To start painting in the Scene, first create a new Tilemap.");
        private bool needCreate
        {
            get => GridPaintingState.scenePaintTarget == null
                   && (GridPaintingState.validTargets == null
                       || GridPaintingState.validTargets.Length == 0);
        }

        private readonly VisualElement m_CreateTargetElement;

        private static string[] GetTilePaletteOverlayIds()
        {
            var types = TypeCache.GetTypesWithAttribute(typeof(TilePaletteOverlayToolbarElementAttribute));
            var overlayIds = new string[types.Count];
            var i = 0;
            foreach (var type in types)
            {
                var toolbarElementAttribute = Attribute.GetCustomAttribute(type, typeof(EditorToolbarElementAttribute)) as EditorToolbarElementAttribute;
                if (toolbarElementAttribute != null)
                    overlayIds[i++] = toolbarElementAttribute.id;
            }
            Array.Resize(ref overlayIds, i);
            return overlayIds;
        }

        public GridPaintPaletteWindowActiveTargets()
        {
            AddToClassList(kUssClassName);
            TilePaletteOverlayUtility.SetStyleSheet(this);

            name = "activeTargetsTilePalette";

            var activeTargets = new VisualElement();
            activeTargets.AddToClassList(kPopupUssClassName);
            activeTargets.Add(new TilePaletteActiveTargetsPopupIcon());
            activeTargets.Add(new TilePaletteActiveTargetsPopup());

            var rightToolbar = EditorToolbar.CreateOverlay(GetTilePaletteOverlayIds());
            rightToolbar.SetupChildrenAsButtonStrip();
            activeTargets.Add(rightToolbar);
            Add(activeTargets);

            m_CreateTargetElement = new VisualElement();
            m_CreateTargetElement.AddToClassList(kInfoUssClassName);

            var createTargetIconElement = new VisualElement();
            createTargetIconElement.name = "Create Target Icon";
            createTargetIconElement.AddToClassList(kCreateHintUssClassName);
            createTargetIconElement.tooltip = k_CreateTargetTooltip;
            m_CreateTargetElement.Add(createTargetIconElement);

            var createTargetLabelElement = new Label(k_CreateTargetTooltip);
            m_CreateTargetElement.Add(createTargetLabelElement);

            Add(m_CreateTargetElement);

            RegisterCallback<AttachToPanelEvent>(OnAttachedToPanel);
            RegisterCallback<DetachFromPanelEvent>(OnDetachFromPanel);
        }

        private void OnAttachedToPanel(AttachToPanelEvent evt)
        {
            GridPaintingState.scenePaintTargetChanged += OnScenePaintTargetChanged;
            GridPaintingState.validTargetsChanged += UpdateCreateTarget;
        }

        private void OnDetachFromPanel(DetachFromPanelEvent evt)
        {
            GridPaintingState.scenePaintTargetChanged -= OnScenePaintTargetChanged;
            GridPaintingState.validTargetsChanged -= UpdateCreateTarget;
        }

        private void OnScenePaintTargetChanged(GameObject go)
        {
            UpdateCreateTarget();
        }

        private void UpdateCreateTarget()
        {
            m_CreateTargetElement.visible = needCreate;
            m_CreateTargetElement.style.position = needCreate ? Position.Relative : Position.Absolute;
        }
    }
}
