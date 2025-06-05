using System;

using UnityEngine;

namespace Unity.PlasticSCM.Editor.UI.Tree
{
    internal class EmptyStateData
    {
        internal Rect Rect { get { return mLastValidRect; } }
        internal GUIContent Content { get { return mContent; } }

        internal bool IsEmpty()
        {
            return string.IsNullOrEmpty(mContent.text);
        }

        internal void Update(string contentText, Rect rect, EventType eventType, Action repaint)
        {
            UpdateText(contentText);

            UpdateValidRect(rect, eventType, repaint);
        }

        internal void UpdateText(string contentText)
        {
            mContent.text = contentText;
        }

        internal void UpdateValidRect(Rect rect, EventType eventType, Action repaint)
        {
            mLastValidRect = EnsureValidRect(rect, mLastValidRect, eventType, repaint);
        }

        internal static Rect EnsureValidRect(
            Rect rect, Rect lastValidRect, EventType eventType, Action repaint)
        {
            if (eventType == EventType.Layout)
                return lastValidRect;

            if (lastValidRect == rect)
                return lastValidRect;

            // Unity's layout system initially provides a placeholder rectangle during Layout.
            // A valid rectangle is only provided on following events like Repaint or Mouse events.
            //
            // - If we use the placeholder rectangle, the layout system won't position UI elements correctly.
            // - If we skip layout processing when the rectangle is invalid, we break GUILayout’s Begin/End pairing.
            //
            // To prevent both issues, we save the last valid rectangle and use it for drawing.

            repaint();

            return rect;
        }

        Rect mLastValidRect;

        readonly GUIContent mContent = new GUIContent(string.Empty);
    }

    internal static class DrawTreeViewEmptyState
    {
        internal static void For(EmptyStateData data)
        {
            DrawCenteredOnRect(data.Rect, ()=>
            {
                GUILayout.Label(data.Content, UnityStyles.Tree.StatusLabel);
            });
        }

        internal static void For(Texture2D icon, EmptyStateData data)
        {
            DrawCenteredOnRect(data.Rect, () =>
            {
                DrawIconAndLabel(icon, data.Content);
            });
        }

        internal static void DrawCenteredOnRect(Rect rect, Action onGUI)
        {
            GUILayout.BeginArea(rect);

            GUILayout.FlexibleSpace();

            GUILayout.BeginHorizontal();
            GUILayout.FlexibleSpace();

            onGUI.Invoke();

            GUILayout.FlexibleSpace();
            GUILayout.EndHorizontal();

            GUILayout.FlexibleSpace();

            GUILayout.EndArea();
        }

        static void DrawIconAndLabel(Texture2D icon, GUIContent label)
        {
            GUILayout.Label(
                icon,
                UnityStyles.Tree.StatusLabel,
                GUILayout.Width(UnityConstants.TREEVIEW_STATUS_ICON_SIZE),
                GUILayout.Height(UnityConstants.TREEVIEW_STATUS_ICON_SIZE));
            GUILayout.Space(UnityConstants.TREEVIEW_STATUS_CONTENT_PADDING);
            GUILayout.Label(label, UnityStyles.Tree.StatusLabel);
        }
    }
}
