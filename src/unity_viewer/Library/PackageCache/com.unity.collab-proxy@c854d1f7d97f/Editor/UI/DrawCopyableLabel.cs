using UnityEditor;
using UnityEngine;

using PlasticGui;

namespace Unity.PlasticSCM.Editor.UI
{
    internal static class DrawCopyableLabel
    {
        internal static void For(string label, GUIStyle style)
        {
            Rect rect = GUILayoutUtility.GetRect(
                new GUIContent(label), style);

            GUI.Label(rect, label, style);

            if (Event.current.type != EventType.ContextClick)
                return;

            if (!rect.Contains(Event.current.mousePosition))
                return;

            GenericMenu menu = new GenericMenu();
            menu.AddItem(
                new GUIContent(PlasticLocalization.Name.Copy.GetString()),
                false,
                () => EditorGUIUtility.systemCopyBuffer = label);
            menu.ShowAsContext();

            Event.current.Use();
        }
    }
}
