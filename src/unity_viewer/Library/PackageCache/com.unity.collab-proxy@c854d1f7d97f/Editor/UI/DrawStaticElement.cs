using UnityEngine;

namespace Unity.PlasticSCM.Editor.UI
{
    internal static class DrawStaticElement
    {
        internal static void Empty()
        {
            GUILayout.Label(GUIContent.none, UnityStyles.NoSizeStyle);
        }
    }
}
