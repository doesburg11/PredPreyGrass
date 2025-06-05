using System;
using UnityEditor.EditorTools;
using UnityEditor.ShortcutManagement;
using UnityEditor.Toolbars;
using UnityEngine;
using UnityEngine.UIElements;

namespace UnityEditor.Tilemaps
{
    internal class TilemapEditorToolButton : EditorToolbarToggle
    {
        private TilemapEditorTool m_TilemapEditorTool;

        public TilemapEditorToolButton(TilemapEditorTool tool)
        {
            focusable = false;

            if (tool != null)
            {
                name = tool.name;
                icon = tool.toolbarIcon?.image as Texture2D;
                tooltip = tool.toolbarIcon?.tooltip;
                m_TilemapEditorTool = tool;
            }

            this.RegisterValueChangedCallback((evt) =>
            {
                SetToolActive();
            });

            RegisterCallback<AttachToPanelEvent>(OnAttachedToPanel);
            RegisterCallback<DetachFromPanelEvent>(OnDetachFromPanel);

            UpdateState();
        }

        private void OnAttachedToPanel(AttachToPanelEvent evt)
        {
            ToolManager.activeToolChanged += UpdateState;
            ToolManager.activeContextChanged += UpdateState;
            ShortcutIntegration.instance.profileManager.shortcutBindingChanged += UpdateTooltips;
            UpdateState();
        }

        private void OnDetachFromPanel(DetachFromPanelEvent evt)
        {
            ShortcutIntegration.instance.profileManager.shortcutBindingChanged -= UpdateTooltips;
            ToolManager.activeToolChanged -= UpdateState;
            ToolManager.activeContextChanged -= UpdateState;
        }

        protected void SetToolActive()
        {
            var active = EditorToolManager.activeTool;
            if (active == m_TilemapEditorTool)
                ToolManager.RestorePreviousPersistentTool();
            else
                ToolManager.SetActiveTool(m_TilemapEditorTool);
            UpdateState();
        }

        private void UpdateState()
        {
            var activeTool = m_TilemapEditorTool == EditorToolManager.activeTool;
            SetValueWithoutNotify(activeTool);
        }

        private void UpdateTooltips(IShortcutProfileManager arg1, Identifier arg2, ShortcutBinding arg3, ShortcutBinding arg4)
        {
            tooltip = m_TilemapEditorTool != null ? m_TilemapEditorTool.toolbarIcon.tooltip : String.Empty;
        }
    }
}
