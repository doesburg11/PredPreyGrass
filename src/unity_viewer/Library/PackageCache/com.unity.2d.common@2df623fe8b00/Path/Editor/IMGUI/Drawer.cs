using System.Collections.Generic;
using UnityEngine;
using UnityEditor;
using UnityEditor.U2D.Common.Path.GUIFramework;

namespace UnityEditor.U2D.Common.Path
{
    internal class Drawer : IDrawer
    {
        internal class Styles
        {
            public readonly GUIStyle pointNormalStyle;
            public readonly GUIStyle tangentNormalStyle;
            public readonly GUIStyle tangentHoveredStyle;

            public Styles()
            {
                var pointNormal = AssetDatabase.LoadAssetAtPath<Texture2D>("Packages/com.unity.2d.common/Path/Editor/Handles/Path/pointNormal.png");
                pointNormalStyle = CreateStyle(pointNormal, Vector2.one * 12f);
                tangentNormalStyle = CreateStyle(pointNormal, Vector2.one * 8f);
                tangentHoveredStyle = CreateStyle(pointNormal, Vector2.one * 10f);
            }

            private GUIStyle CreateStyle(Texture2D texture, Vector2 size)
            {
                var guiStyle = new GUIStyle();
                guiStyle.normal.background = texture;
                guiStyle.fixedWidth = size.x;
                guiStyle.fixedHeight = size.y;

                return guiStyle;
            }
        }

        private IGUIState m_GUIState = new GUIState();
        private Styles m_Styles;
        private Styles styles
        {
            get
            {
                if (m_Styles == null)
                    m_Styles = new Styles();

                return m_Styles;
            }
        }

        public void DrawCreatePointPreview(Vector3 position, Color color)
        {
            Color saved = GUI.color;
            GUI.color = color;            
            DrawGUIStyleCap(0, position, Quaternion.identity, m_GUIState.GetHandleSize(position), styles.pointNormalStyle);
            GUI.color = saved;
        }

        public void DrawPoint(Vector3 position, Color color)
        {
            Color saved = GUI.color;
            GUI.color = color;
            DrawGUIStyleCap(0, position, Quaternion.identity, m_GUIState.GetHandleSize(position), styles.pointNormalStyle);
            GUI.color = saved;
        }

        public void DrawPointHovered(Vector3 position, Color color)
        {
            Color saved = GUI.color;
            GUI.color = color;            
            DrawGUIStyleCap(0, position, Quaternion.identity, m_GUIState.GetHandleSize(position), styles.pointNormalStyle);
            GUI.color = saved;
        }

        public void DrawPointSelected(Vector3 position, Color color)
        {
            Color saved = GUI.color;
            GUI.color = color;            
            DrawGUIStyleCap(0, position, Quaternion.identity, m_GUIState.GetHandleSize(position), styles.pointNormalStyle);
            GUI.color = saved;
        }

        public void DrawLine(Vector3 p1, Vector3 p2, float width, Color color)
        {
            Handles.color = color;
            Handles.DrawAAPolyLine(width, new Vector3[] { p1, p2 });
        }

        public void DrawBezier(Vector3 p1, Vector3 p2, Vector3 p3, Vector3 p4, float width, Color color)
        {
            Handles.color = color;
            Handles.DrawBezier(p1, p4, p2, p3, color, null, width);
        }

        public void DrawTangent(Vector3 position, Vector3 tangent, Color color)
        {
            DrawLine(position, tangent, 3f, color);
            Color saved = GUI.color;
            GUI.color = color;
            DrawGUIStyleCap(0, tangent, Quaternion.identity, m_GUIState.GetHandleSize(tangent), styles.tangentNormalStyle);
            GUI.color = saved;
        }

        
        private void DrawGUIStyleCap(int controlID, Vector3 position, Quaternion rotation, float size, GUIStyle guiStyle)
        {
            if (Camera.current && Vector3.Dot(position - Camera.current.transform.position, Camera.current.transform.forward) < 0f)
                return;

            Handles.BeginGUI();
            guiStyle.Draw(GetGUIStyleRect(guiStyle, position), GUIContent.none, controlID);
            Handles.EndGUI();
        }

        private Rect GetGUIStyleRect(GUIStyle style, Vector3 position)
        {
            Vector2 vector = HandleUtility.WorldToGUIPoint(position);

            float fixedWidth = style.fixedWidth;
            float fixedHeight = style.fixedHeight;

            return new Rect(vector.x - fixedWidth / 2f, vector.y - fixedHeight / 2f, fixedWidth, fixedHeight);
        }
    }
}
