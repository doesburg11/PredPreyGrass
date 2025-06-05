using System.IO;
using UnityEngine;

namespace UnityEditor.U2D.Animation
{
    internal static class EditorIconUtility
    {
        public const string LightIconPath = "EditorIcons/Light";
        public const string DarkIconPath = "EditorIcons/Dark";

        public static Texture2D LoadIconResource(string name, string lightPath, string darkPath)
        {
            var iconPath = "";

            if (EditorGUIUtility.isProSkin && !string.IsNullOrEmpty(darkPath))
                iconPath = Path.Combine(darkPath, "d_" + name);
            else
                iconPath = Path.Combine(lightPath, name);
            if (EditorGUIUtility.pixelsPerPoint > 1.0f)
            {
                var icon2x = ResourceLoader.Load<Texture2D>(iconPath + "@2x.png");
                if (icon2x != null)
                    return icon2x;
            }

            return ResourceLoader.Load<Texture2D>(iconPath+".png");
        }
    }
}
