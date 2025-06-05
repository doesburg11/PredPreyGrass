using UnityEngine;

namespace UnityEditor.Tilemaps
{
    internal class TilePaletteMouseCursorUtility
    {
        internal static class MouseStyles
        {
            // The following paths match the enums in OperatingSystemFamily
            public static readonly string[] mouseCursorOSPath =
            {
                "", // Other OS
                "Cursors/macOS",
                "Cursors/Windows",
                "Cursors/Linux",
            };
            // The following paths match the enums in OperatingSystemFamily
            public static readonly Vector2[] mouseCursorOSHotspot =
            {
                Vector2.zero, // Other OS
                new Vector2(6f, 4f),
                new Vector2(6f, 4f),
                new Vector2(6f, 4f),
            };
            // The following paths match the enums in sceneViewEditModes above
            public static readonly string[] mouseCursorTexturePaths =
            {
                "",
                "Grid.MoveTool.png",
                "Grid.PaintTool.png",
                "Grid.BoxTool.png",
                "Grid.PickingTool.png",
                "Grid.EraserTool.png",
                "Grid.FillTool.png",
            };
            public static readonly Texture2D[] mouseCursorTextures;

            public static readonly GridBrushBase.Tool[] sceneViewEditModes =
            {
                GridBrushBase.Tool.Select,
                GridBrushBase.Tool.Move,
                GridBrushBase.Tool.Paint,
                GridBrushBase.Tool.Box,
                GridBrushBase.Tool.Pick,
                GridBrushBase.Tool.Erase,
                GridBrushBase.Tool.FloodFill
            };

            static MouseStyles()
            {
                mouseCursorTextures = new Texture2D[mouseCursorTexturePaths.Length];
                int osIndex = (int)SystemInfo.operatingSystemFamily;
                for (int i = 0; i < mouseCursorTexturePaths.Length; ++i)
                {
                    if ((mouseCursorOSPath[osIndex] != null && mouseCursorOSPath[osIndex].Length > 0)
                        && (mouseCursorTexturePaths[i] != null && mouseCursorTexturePaths[i].Length > 0))
                    {
                        string cursorPath = Utils.Paths.Combine(mouseCursorOSPath[osIndex], mouseCursorTexturePaths[i]);
                        mouseCursorTextures[i] = EditorGUIUtility.LoadRequired(cursorPath) as Texture2D;
                    }
                    else
                        mouseCursorTextures[i] = null;
                }
            }
        }
    }
}
