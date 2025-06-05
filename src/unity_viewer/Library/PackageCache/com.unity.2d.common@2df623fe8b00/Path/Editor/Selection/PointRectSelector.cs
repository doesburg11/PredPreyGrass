using UnityEngine;
using UnityEditor;

namespace UnityEditor.U2D.Common.Path
{
    internal class PointRectSelector : RectSelector<Vector3>
    {
        protected override bool Select(Vector3 element)
        {
            return guiRect.Contains(HandleUtility.WorldToGUIPoint(element), true);
        }
    }
}
