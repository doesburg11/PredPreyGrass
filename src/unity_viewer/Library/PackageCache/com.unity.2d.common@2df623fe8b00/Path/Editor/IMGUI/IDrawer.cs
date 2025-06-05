using UnityEngine;

namespace UnityEditor.U2D.Common.Path
{
    internal interface IDrawer
    {
        void DrawCreatePointPreview(Vector3 position, Color color);
        void DrawPoint(Vector3 position, Color color);
        void DrawPointHovered(Vector3 position, Color color);
        void DrawPointSelected(Vector3 position, Color color);
        void DrawLine(Vector3 p1, Vector3 p2, float width, Color color);
        void DrawBezier(Vector3 p1, Vector3 p2, Vector3 p3, Vector3 p4, float width, Color color);
        void DrawTangent(Vector3 position, Vector3 tangent, Color color);
    }
}
