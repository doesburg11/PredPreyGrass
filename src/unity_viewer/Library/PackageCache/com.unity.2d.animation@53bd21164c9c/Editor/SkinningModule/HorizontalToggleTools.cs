using System;
using UnityEngine;

namespace UnityEditor.U2D.Animation
{
    internal class HorizontalToggleTools
    {
        private static class Styles
        {
            public static GUIContent visibilityCollapseIcon = new GUIContent(EditorIconUtility.LoadIconResource("Visibility_Tool", EditorIconUtility.LightIconPath, EditorIconUtility.DarkIconPath), TextContent.visibilityIconTooltip);
            public static GUIContent visibilityIcon = new GUIContent(TextContent.visibilityIconText, EditorIconUtility.LoadIconResource("Visibility_Tool", EditorIconUtility.LightIconPath, EditorIconUtility.DarkIconPath), TextContent.visibilityIconTooltip);
            public static GUIContent spriteSheetIcon = new GUIContent(TextContent.spriteSheetIconText, EditorIconUtility.LoadIconResource("Sprite_Mode", EditorIconUtility.LightIconPath, EditorIconUtility.DarkIconPath), TextContent.spriteSheetIconTooltip);
            public static GUIContent spriteSheetCollapseIcon = new GUIContent(EditorIconUtility.LoadIconResource("Sprite_Mode", EditorIconUtility.LightIconPath, EditorIconUtility.DarkIconPath), TextContent.spriteSheetIconTooltip);
        }

        private SkinningCache skinningCache { get; set; }

        private VisibilityTool visibilityTool
        {
            get { return skinningCache.GetTool(Tools.Visibility) as VisibilityTool; }
        }

        private SwitchModeTool switchmodeTool
        {
            get { return skinningCache.GetTool(Tools.SwitchMode) as SwitchModeTool; }
        }

        private GUIContent spriteSheetIcon
        {
            get { return collapseToolbar ? Styles.spriteSheetCollapseIcon : Styles.spriteSheetIcon; }
        }

        internal Action<BaseTool> onActivateTool = (b) => { };

        public bool collapseToolbar { get; set; }

        internal HorizontalToggleTools(SkinningCache s)
        {
            skinningCache = s;
        }

        internal void DoGUI(Rect drawArea, BaseTool currentTool, bool isDisabled)
        {
            using (new EditorGUI.DisabledScope(isDisabled))
            {
                GUILayout.BeginArea(drawArea);
                EditorGUILayout.BeginHorizontal();

                DoModeToggle();
                GUILayout.FlexibleSpace();
                DoVisibilityToggle(currentTool);

                EditorGUILayout.EndHorizontal();
                GUILayout.EndArea();
            }
        }

        private void DoModeToggle()
        {
            if (skinningCache.hasCharacter)
            {
                EditorGUI.BeginChangeCheck();
                var isActive = GUILayout.Toggle(switchmodeTool.isActive, spriteSheetIcon, EditorStyles.toolbarButton);
                if (EditorGUI.EndChangeCheck())
                {
                    using (skinningCache.UndoScope(TextContent.setMode))
                    {
                        if (isActive)
                            switchmodeTool.Activate();
                        else
                            switchmodeTool.Deactivate();
                    }
                }
            }
        }

        void DoVisibilityToggle(BaseTool currentTool)
        {
            EditorGUI.BeginChangeCheck();
            GUILayout.Toggle(visibilityTool.isActive, visbilityIcon, EditorStyles.toolbarButton);
            if (EditorGUI.EndChangeCheck())
                ToggleVisibilityTool(currentTool);
        }

        GUIContent visbilityIcon => collapseToolbar ? Styles.visibilityCollapseIcon : Styles.visibilityIcon;

        internal void ToggleVisibilityTool(BaseTool currentTool)
        {
            onActivateTool(visibilityTool);
        }
    }
}
