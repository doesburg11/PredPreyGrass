using System;
using System.Text.RegularExpressions;

using UnityEditor;
using UnityEngine;

using Codice.Client.Common;

namespace Unity.PlasticSCM.Editor.UI
{
    internal static class DrawTextBlockWithLink
    {
        internal static void ForExternalLink(
            ExternalLink externalLink,
            string explanation,
            GUIStyle textBlockStyle)
        {
            GUILayout.Label(explanation, textBlockStyle);

            GUIStyle linkStyle = new GUIStyle(UnityStyles.LinkLabel);
            linkStyle.fontSize = textBlockStyle.fontSize;
            linkStyle.stretchWidth = false;

            if (GUILayout.Button(externalLink.Label, linkStyle))
                Application.OpenURL(externalLink.Url);

            EditorGUIUtility.AddCursorRect(
                GUILayoutUtility.GetLastRect(), MouseCursor.Link);
        }

        internal static void ForMultiLinkLabelInDialog(
            MultiLinkLabelData data)
        {
            ForMultiLinkLabel(
                data,
                UnityStyles.Paragraph,
                areLinkActionsSupported: Application.platform == RuntimePlatform.WindowsEditor,
                isEndFlexibleSpaceNeeded: true);
        }

        internal static void ForMultiLinkLabel(
            MultiLinkLabelData data,
            GUIStyle style)
        {
            ForMultiLinkLabel(
                data, style,
                areLinkActionsSupported: true,
                isEndFlexibleSpaceNeeded: false);
        }

        static void ForMultiLinkLabel(
            MultiLinkLabelData data,
            GUIStyle style,
            bool areLinkActionsSupported,
            bool isEndFlexibleSpaceNeeded)
        {
            GUIStyle labelStyle = new GUIStyle(style);
            labelStyle.margin = new RectOffset(0, 0, style.margin.top, style.margin.bottom);
            labelStyle.padding = new RectOffset(0, 0, style.padding.top, style.padding.bottom);

            if (!areLinkActionsSupported)
            {
                GUILayout.Label(string.Format(data.Text, data.LinkNames.ToArray()), labelStyle);
                return;
            }

            string[] labels = Regex.Split(data.Text, @"\{\d+\}");

            GUIStyle linkStyle = new GUIStyle(labelStyle);
            linkStyle.normal.textColor = EditorStyles.linkLabel.normal.textColor;
            linkStyle.hover.textColor = EditorStyles.linkLabel.hover.textColor;
            linkStyle.active.textColor = EditorStyles.linkLabel.active.textColor;
            linkStyle.stretchWidth = false;

            using (new EditorGUILayout.HorizontalScope())
            {
                for (int i = 0; i < labels.Length; i++)
                {
                    GUILayout.Label(labels[i], labelStyle);

                    if (data.LinkNames.Count <= i)
                        break;

                    bool buttonResult = GUILayout.Button(data.LinkNames[i], linkStyle);

                    EditorGUIUtility.AddCursorRect(
                        GUILayoutUtility.GetLastRect(), MouseCursor.Link);

                    if (buttonResult)
                        ((Action)data.LinkActions[i]).Invoke();
                }

                if (isEndFlexibleSpaceNeeded)
                    GUILayout.FlexibleSpace();
            }
        }
    }
}
