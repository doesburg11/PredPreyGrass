using UnityEngine.UIElements;

namespace UnityEditor.Tilemaps
{
    internal static class TilePaletteOverlayUtility
    {
        private static readonly string s_TilePaletteOverlayStyleSheetPath = "Packages/com.unity.2d.tilemap/Editor/UI/TilePaletteOverlay.uss";
        private static readonly string s_TilePaletteOverlayStyleSheetShiroPath = "Packages/com.unity.2d.tilemap/Editor/UI/TilePaletteOverlayLight.uss";
        private static readonly string s_TilePaletteOverlayStyleSheetKuroPath = "Packages/com.unity.2d.tilemap/Editor/UI/TilePaletteOverlayDark.uss";

        private static StyleSheet s_TilePaletteOverlayStyleSheet;
        private static StyleSheet s_TilePaletteOverlayStyleSheetLight;
        private static StyleSheet s_TilePaletteOverlayStyleSheetDark;

        private static readonly string buttonStripClassName = "unity-editor-toolbar__button-strip";
        private static readonly string stripElementClassName = buttonStripClassName + "-element";
        private static readonly string leftStripElementClassName = stripElementClassName + "--left";
        private static readonly string middleStripElementClassName = stripElementClassName + "--middle";
        private static readonly string rightStripElementClassName = stripElementClassName + "--right";
        private static readonly string aloneStripElementClassName = stripElementClassName + "--alone";
        private static StyleSheet StyleSheet
        {
            get
            {
                if (s_TilePaletteOverlayStyleSheet == null)
                    s_TilePaletteOverlayStyleSheet = EditorGUIUtility.Load(s_TilePaletteOverlayStyleSheetPath) as StyleSheet;
                return s_TilePaletteOverlayStyleSheet;
            }
        }

        private static StyleSheet StyleSheetLight
        {
            get
            {
                if (s_TilePaletteOverlayStyleSheetLight == null)
                    s_TilePaletteOverlayStyleSheetLight = EditorGUIUtility.Load(s_TilePaletteOverlayStyleSheetShiroPath) as StyleSheet;
                return s_TilePaletteOverlayStyleSheetLight;
            }
        }

        private static StyleSheet StyleSheetDark
        {
            get
            {
                if (s_TilePaletteOverlayStyleSheetDark == null)
                    s_TilePaletteOverlayStyleSheetDark = EditorGUIUtility.Load(s_TilePaletteOverlayStyleSheetKuroPath) as StyleSheet;
                return s_TilePaletteOverlayStyleSheetDark;
            }
        }

        internal static void SetStyleSheet(VisualElement ve)
        {
            ve.styleSheets.Add(StyleSheet);
            if (EditorGUIUtility.isProSkin)
                ve.styleSheets.Add(StyleSheetDark);
            else
                ve.styleSheets.Add(StyleSheetLight);
        }

        internal static void SetupChildrenAsButtonStripForVisible(VisualElement root, bool[] visibleList)
        {
            root.AddToClassList(buttonStripClassName);

            var count = root.hierarchy.childCount;
            if (count != visibleList.Length)
                return;

            if (count == 1)
            {
                var element = root.hierarchy.ElementAt(0);
                var visible = visibleList[0];
                element.EnableInClassList(aloneStripElementClassName, visible);
                if (visible)
                {
                    element.style.position = Position.Relative;
                    element.style.visibility = Visibility.Visible;
                }
                else
                {
                    element.style.position = Position.Absolute;
                    element.style.visibility = Visibility.Hidden;
                }
            }
            else
            {
                int lastVisible = 0;
                bool firstVisible = true;
                for (var i = 0; i < count; ++i)
                {
                    var element = root.hierarchy.ElementAt(i);
                    var visible = visibleList[i];

                    element.AddToClassList(stripElementClassName);
                    element.RemoveFromClassList(leftStripElementClassName);
                    element.RemoveFromClassList(middleStripElementClassName);
                    element.RemoveFromClassList(rightStripElementClassName);

                    if (firstVisible)
                    {
                        element.EnableInClassList(leftStripElementClassName, visible);
                        firstVisible = false;
                    }
                    else
                    {
                        element.EnableInClassList(middleStripElementClassName, visible);
                        element.RemoveFromClassList(rightStripElementClassName);
                    }

                    if (visible)
                    {
                        lastVisible = i;
                        element.style.position = Position.Relative;
                        element.style.visibility = Visibility.Visible;
                    }
                    else
                    {
                        element.style.position = Position.Absolute;
                        element.style.visibility = Visibility.Hidden;
                    }
                }
                var lastElement = root.hierarchy.ElementAt(lastVisible);
                if (lastElement.ClassListContains(leftStripElementClassName))
                {
                    lastElement.RemoveFromClassList(leftStripElementClassName);
                    lastElement.AddToClassList(aloneStripElementClassName);
                }
                else
                {
                    lastElement.RemoveFromClassList(middleStripElementClassName);
                    lastElement.AddToClassList(rightStripElementClassName);
                }
            }
        }
    }
}
