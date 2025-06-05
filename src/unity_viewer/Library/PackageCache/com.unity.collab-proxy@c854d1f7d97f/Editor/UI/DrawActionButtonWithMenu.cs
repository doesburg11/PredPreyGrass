using System;

using UnityEditor;
using UnityEngine;

namespace Unity.PlasticSCM.Editor.UI
{
    internal static class DrawActionButtonWithMenu
    {
        internal static GUIStyle ButtonStyle { get { return mButtonStyle; } }

        internal static void For(
            string text,
            string tooltip,
            Action buttonAction,
            GenericMenu actionMenu)
        {
            float width = MeasureMaxWidth.ForTexts(mButtonStyle, text);

            For(text, tooltip, width, buttonAction, actionMenu);
        }

        internal static void For(
            string text,
            string tooltip,
            float width,
            Action buttonAction,
            GenericMenu actionMenu)
        {
            // Action button
            GUIContent buttonContent = new GUIContent(text, tooltip);

            Rect rt = GUILayoutUtility.GetRect(
                buttonContent,
                mButtonStyle,
                GUILayout.MinWidth(width),
                GUILayout.MaxWidth(width));

            if (GUI.Button(rt, buttonContent, mButtonStyle))
            {
                buttonAction();
            }

            // Menu dropdown
            GUIContent dropDownContent = new GUIContent(
                string.Empty, Images.GetDropDownIcon());

            Rect dropDownRect = GUILayoutUtility.GetRect(
                dropDownContent,
                mDropDownStyle,
                GUILayout.MinWidth(DROPDOWN_BUTTON_WIDTH),
                GUILayout.MaxWidth(DROPDOWN_BUTTON_WIDTH));

            if (EditorGUI.DropdownButton(
                    dropDownRect, dropDownContent, FocusType.Passive, mDropDownStyle))
            {
                actionMenu.DropDown(dropDownRect);
            }
        }

        static readonly GUIStyle mButtonStyle =
            new GUIStyle(EditorStyles.miniButtonLeft)
            {
                stretchWidth = false
            };

        static readonly GUIStyle mDropDownStyle =
            new GUIStyle(EditorStyles.miniButtonRight);

        const int DROPDOWN_BUTTON_WIDTH = 16;
    }
}
