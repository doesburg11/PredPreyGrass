using UnityEditor;
using UnityEngine;

namespace Unity.PlasticSCM.Editor.UI
{
    internal static class DrawActionButton
    {
        internal static GUIStyle ButtonStyle { get { return mButtonStyle; } }

        internal static bool For(string buttonText)
        {
            GUIContent buttonContent = new GUIContent(buttonText);

            return ForRegularButton(buttonContent);
        }

        internal static bool For(string buttonText, string buttonTooltip)
        {
            GUIContent buttonContent = new GUIContent(buttonText, buttonTooltip);

            return ForRegularButton(buttonContent);
        }

        internal static bool ForCommentSection(string buttonText, float width)
        {
            GUIContent buttonContent = new GUIContent(buttonText);

            Rect rt = GUILayoutUtility.GetRect(
                buttonContent,
                mButtonStyle,
                GUILayout.MinWidth(width),
                GUILayout.MaxWidth(width));

            return GUI.Button(rt, buttonContent, mButtonStyle);
        }

        static bool ForRegularButton(GUIContent buttonContent)
        {
            Rect rt = GUILayoutUtility.GetRect(
                buttonContent,
                mButtonStyle,
                GUILayout.MinWidth(UnityConstants.REGULAR_BUTTON_WIDTH));

            return GUI.Button(rt, buttonContent, mButtonStyle);
        }

        static readonly GUIStyle mButtonStyle =
            new GUIStyle(EditorStyles.miniButton)
            {
                stretchWidth = false
            };
    }
}
