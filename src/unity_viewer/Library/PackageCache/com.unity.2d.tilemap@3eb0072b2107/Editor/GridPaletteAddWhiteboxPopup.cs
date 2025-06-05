using UnityEditor.PackageManager.UI;
using UnityEngine;
using UnityEngine.UIElements;

namespace UnityEditor.Tilemaps
{
    internal class GridPaletteAddWhiteboxPopup : EditorWindow
    {
        static class Styles
        {
            public static readonly GUIContent nameLabel = EditorGUIUtility.TrTextContent("Name");
            public static readonly GUIContent ok = EditorGUIUtility.TrTextContent("Create");
            public static readonly GUIContent cancel = EditorGUIUtility.TrTextContent("Cancel");
            public static readonly GUIContent header = EditorGUIUtility.TrTextContent("Create New Palette");
            public static readonly GUIContent gridLabel = EditorGUIUtility.TrTextContent("Grid");
            public static readonly GUIContent sizeLabel = EditorGUIUtility.TrTextContent("Cell Size");
            public static readonly GUIContent hexagonLabel = EditorGUIUtility.TrTextContent("Hexagon Type");
        }

        private static long s_LastClosedTime;
        private static GridPaletteAddWhiteboxPopup s_Instance;

        private DropdownField m_DropdownField;


        void Init(Rect buttonRect)
        {
            buttonRect = GUIUtility.GUIToScreenRect(buttonRect);
            ShowAsDropDown(buttonRect, new Vector2(380, 45));
        }

        internal void CreateGUI()
        {
            var root = new Box();
            root.style.borderBottomWidth = 1f;
            root.style.borderTopWidth = 1f;
            root.style.borderLeftWidth = 1f;
            root.style.borderRightWidth = 1f;
            root.style.borderBottomColor = Color.gray;
            root.style.borderTopColor = Color.gray;
            root.style.borderLeftColor = Color.gray;
            root.style.borderRightColor = Color.gray;
            rootVisualElement.Add(root);

            m_DropdownField = new DropdownField("White Box Type", TilePaletteWhiteboxSamplesUtility.whiteboxSampleNames, 0);
            root.Add(m_DropdownField);

            var buttons = new VisualElement();
            buttons.style.flexDirection = FlexDirection.Row;

            var importButton = new Button(Import);
            importButton.text = "Import";
            var space = new VisualElement();
            space.style.flexGrow = 1f;
            var cancelButton = new Button(Cancel);
            cancelButton.text = "Cancel";

            buttons.Add(space);
            buttons.Add(importButton);
            buttons.Add(cancelButton);

            root.Add(buttons);
        }

        private void Import()
        {
            TilePaletteWhiteboxSamplesUtility.ImportWhiteboxSample(m_DropdownField.index);
            if (GridPaintPaletteClipboard.instances is { Count: > 0 })
            {
                var clipboard = GridPaintPaletteClipboard.instances[0];
                clipboard.PickFirstFromPalette();
            }
            Close();
        }

        private void Cancel()
        {
            Close();
        }

        internal static bool ShowAtPosition(Rect buttonRect)
        {
            // We could not use realtimeSinceStartUp since it is set to 0 when entering/exitting playmode, we assume an increasing time when comparing time.
            long nowMilliSeconds = System.DateTime.Now.Ticks / System.TimeSpan.TicksPerMillisecond;
            bool justClosed = nowMilliSeconds < s_LastClosedTime + 50;
            if (!justClosed)
            {
                Event.current.Use();
                if (s_Instance == null)
                    s_Instance = ScriptableObject.CreateInstance<GridPaletteAddWhiteboxPopup>();

                s_Instance.Init(buttonRect);
                return true;
            }
            return false;
        }
    }
}

// namespace
